"""Entity network (customer / account / device / merchant) and optional directed
transfer-ledger analysis (paths, cycles, pass-through accounts)."""
from __future__ import annotations

from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import COLORS, RISK_COLORS
from .model import band

NODE_STYLE = {
    "customer": dict(color=COLORS["info"], symbol="circle", label="Customer"),
    "account": dict(color=COLORS["navy"], symbol="square", label="Account"),
    "device": dict(color=COLORS["purple"], symbol="diamond", label="Device"),
    "merchant": dict(color=COLORS["teal"], symbol="triangle-up", label="Merchant"),
}
PREFIX = {"customer": "C", "account": "A", "device": "D", "merchant": "M"}
COL = {"customer": "customer_id", "account": "account_id", "device": "device_id", "merchant": "merchant_id"}


def node_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def build_entity_graph(scored: pd.DataFrame) -> nx.Graph:
    """Account-centric association graph: account–customer, account–device,
    account–merchant, plus customer–device (device sharing)."""
    G = nx.Graph()
    for kind, col in COL.items():
        agg = scored.groupby(col).agg(risk=("risk_score", "max"), txns=("transaction_id", "count"),
                                      amount=("amount", "sum"))
        bank = scored.groupby(col)["bank_id"].first() if kind == "customer" else None
        for ent, row in agg.iterrows():
            G.add_node(node_id(kind, ent), kind=kind, entity=ent, risk=float(row.risk), txns=int(row.txns),
                       amount=float(row.amount), bank=bank[ent] if bank is not None else None)
    pairs = [("account", "customer"), ("account", "device"), ("account", "merchant"), ("customer", "device")]
    for a, b in pairs:
        e = scored.groupby([COL[a], COL[b]]).agg(txns=("transaction_id", "count"), amount=("amount", "sum"),
                                                 risk=("risk_score", "max")).reset_index()
        for _, r in e.iterrows():
            G.add_edge(node_id(a, r[COL[a]]), node_id(b, r[COL[b]]), txns=int(r.txns),
                       amount=float(r.amount), risk=float(r.risk))
    return G


def local_graph(G: nx.Graph, center: str, radius: int = 1, max_nodes: int = 60) -> nx.Graph:
    nodes = nx.single_source_shortest_path_length(G, center, cutoff=radius)
    if len(nodes) > max_nodes:
        # keep the centre, then the riskiest nodes closest to it
        ranked = sorted(nodes, key=lambda n: (nodes[n], -G.nodes[n]["risk"]))
        nodes = {n: nodes[n] for n in ranked[:max_nodes]}
    return G.subgraph(nodes).copy()


def entity_figure(H: nx.Graph, center: str, visible_banks: list[str], height: int = 560) -> go.Figure:
    pos = nx.spring_layout(H, seed=7, k=1.1 / np.sqrt(max(len(H), 1)), iterations=80)
    ex, ey = [], []
    for u, v in H.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#CBD5E1", width=1.2),
                               hoverinfo="skip", showlegend=False))
    for kind, style in NODE_STYLE.items():
        ns = [n for n in H if H.nodes[n]["kind"] == kind]
        if not ns:
            continue
        xs, ys, texts, hovers, sizes, lines = [], [], [], [], [], []
        for n in ns:
            d = H.nodes[n]
            masked = kind == "customer" and d.get("bank") not in visible_banks
            name = "External customer" if masked else d["entity"]
            rb = band(d["risk"])
            xs.append(pos[n][0]); ys.append(pos[n][1])
            texts.append("•••" if masked else d["entity"])
            if masked:
                hovers.append("<b>External customer</b><br>Belongs to another bank – details hidden")
            else:
                hovers.append(f"<b>{style['label']} {name}</b><br>Max risk score: {d['risk']:.1f} ({rb})"
                              f"<br>Transactions: {d['txns']}<br>Amount: {d['amount']:,.0f}"
                              f"<br>Connections: {H.degree(n)}")
            sizes.append(34 if n == center else 14 + min(H.degree(n), 10) * 1.6)
            lines.append(COLORS["muted"] if masked else RISK_COLORS[rb])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=style["label"], text=texts, textposition="top center",
            textfont=dict(size=10, color=COLORS["slate"]), hovertext=hovers, hoverinfo="text",
            marker=dict(size=sizes, color=style["color"], symbol=style["symbol"],
                        line=dict(color=lines, width=3)),
        ))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.02, x=0), hoverlabel=dict(bgcolor="#FFFFFF"),
                      dragmode="pan")
    return fig


def neighbours_table(G: nx.Graph, center: str, visible_banks: list[str]) -> pd.DataFrame:
    rows = []
    for n in G.neighbors(center):
        d = G.nodes[n]
        e = G.edges[center, n]
        masked = d["kind"] == "customer" and d.get("bank") not in visible_banks
        rows.append({
            "type": NODE_STYLE[d["kind"]]["label"],
            "entity": "External customer (other bank)" if masked else d["entity"],
            "shared_transactions": e["txns"],
            "shared_amount": round(e["amount"], 2),
            "entity_max_risk": None if masked else round(d["risk"], 1),
            "risk_band": "Hidden" if masked else band(d["risk"]),
        })
    return pd.DataFrame(rows).sort_values(["type", "shared_transactions"], ascending=[True, False]) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- transfer ledger
REQUIRED_LEDGER = ["sender_account", "receiver_account", "amount"]
ALIASES = {
    "sender": "sender_account", "from": "sender_account", "to": "receiver_account",
    "from_account": "sender_account", "source_account": "sender_account",
    "receiver": "receiver_account", "to_account": "receiver_account", "beneficiary_account": "receiver_account",
    "value": "amount", "time": "timestamp", "transaction_time": "timestamp", "date": "timestamp",
}


def validate_ledger(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Return (clean ledger or None, errors, warnings)."""
    errors, warnings = [], []
    df = df.copy()
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    missing = [c for c in REQUIRED_LEDGER if c not in df.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}. "
                      f"A valid ledger needs {', '.join(REQUIRED_LEDGER)} (timestamp optional)."], []
    n = len(df)
    for c in ["sender_account", "receiver_account"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad = df["sender_account"].isna() | df["receiver_account"].isna() | df["amount"].isna() | (df["amount"] <= 0)
    if bad.sum():
        warnings.append(f"{int(bad.sum())} row(s) dropped: missing account IDs or non-positive/invalid amount.")
    df = df[~bad]
    self_t = df["sender_account"] == df["receiver_account"]
    if self_t.sum():
        warnings.append(f"{int(self_t.sum())} self-transfer row(s) dropped.")
    df = df[~self_t]
    dups = df.duplicated()
    if dups.sum():
        warnings.append(f"{int(dups.sum())} exact duplicate transfer(s) dropped.")
    df = df[~dups]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().sum():
            warnings.append(f"{int(df['timestamp'].isna().sum())} unparseable timestamp(s) kept without time.")
    if df.empty:
        errors.append("No valid transfers remain after validation.")
        return None, errors, warnings
    if len(df) < n:
        warnings.append(f"{len(df)} of {n} rows are valid transfers.")
    return df.reset_index(drop=True), errors, warnings


def ledger_graph(ledger: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    agg = ledger.groupby(["sender_account", "receiver_account"]).agg(amount=("amount", "sum"),
                                                                    count=("amount", "size")).reset_index()
    for _, r in agg.iterrows():
        G.add_edge(r.sender_account, r.receiver_account, amount=float(r.amount), count=int(r["count"]))
    return G


def find_cycles(G: nx.DiGraph, max_len: int = 6, limit: int = 200) -> pd.DataFrame:
    rows = []
    for cyc in islice(nx.simple_cycles(G, length_bound=max_len), limit):
        amounts = [G.edges[cyc[i], cyc[(i + 1) % len(cyc)]]["amount"] for i in range(len(cyc))]
        rows.append({"cycle": " → ".join(cyc + [cyc[0]]), "length": len(cyc),
                     "min_edge_amount": round(min(amounts), 2), "total_flow": round(sum(amounts), 2),
                     "accounts": cyc})
    df = pd.DataFrame(rows)
    return df.sort_values(["min_edge_amount"], ascending=False).reset_index(drop=True) if not df.empty else df


def flow_stats(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for n in G.nodes:
        inn = sum(d["amount"] for _, _, d in G.in_edges(n, data=True))
        out = sum(d["amount"] for _, _, d in G.out_edges(n, data=True))
        ratio = min(inn, out) / max(inn, out) if max(inn, out) > 0 else 0
        rows.append({"account": n, "senders": G.in_degree(n), "receivers": G.out_degree(n),
                     "amount_in": round(inn, 2), "amount_out": round(out, 2),
                     "pass_through_ratio": round(ratio, 2),
                     "pass_through_flag": int(inn > 0 and out > 0 and ratio >= 0.8)})
    return pd.DataFrame(rows).sort_values(["pass_through_flag", "amount_in"], ascending=False).reset_index(drop=True)


def find_paths(G: nx.DiGraph, src: str, dst: str, cutoff: int = 6, limit: int = 20) -> pd.DataFrame:
    if src not in G or dst not in G:
        return pd.DataFrame()
    rows = []
    for p in islice(nx.all_simple_paths(G, src, dst, cutoff=cutoff), limit):
        amounts = [G.edges[p[i], p[i + 1]]["amount"] for i in range(len(p) - 1)]
        rows.append({"path": " → ".join(p), "hops": len(p) - 1, "bottleneck_amount": round(min(amounts), 2),
                     "accounts": p})
    df = pd.DataFrame(rows)
    return df.sort_values(["hops", "bottleneck_amount"], ascending=[True, False]).reset_index(drop=True) if not df.empty else df


def ledger_figure(G: nx.DiGraph, highlight: list[str] | None = None, closed: bool = False,
                  max_nodes: int = 80, height: int = 560) -> go.Figure:
    highlight = highlight or []
    H = G
    if len(G) > max_nodes:
        keep = set(highlight)
        for n, _ in sorted(G.degree, key=lambda x: -x[1]):
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        H = G.subgraph(keep)
    pos = nx.spring_layout(H, seed=3, k=1.4 / np.sqrt(max(len(H), 1)), iterations=80)
    hl_edges = set(zip(highlight, highlight[1:]))
    if closed and len(highlight) > 1:
        hl_edges.add((highlight[-1], highlight[0]))
    fig = go.Figure()
    for u, v, d in H.edges(data=True):
        on = (u, v) in hl_edges
        fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.1,
                           arrowwidth=3 if on else 1.2, standoff=9,
                           arrowcolor=COLORS["danger"] if on else "#94A3B8", opacity=1 if on else 0.7)
    hs = set(highlight)
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in H], y=[pos[n][1] for n in H], mode="markers+text",
        text=list(H.nodes), textposition="top center", textfont=dict(size=10),
        hovertext=[f"<b>{n}</b><br>In: {H.in_degree(n)} · Out: {H.out_degree(n)}" for n in H], hoverinfo="text",
        marker=dict(size=[20 if n in hs else 13 for n in H],
                    color=[COLORS["danger"] if n in hs else COLORS["navy"] for n in H],
                    line=dict(color="#FFFFFF", width=1.5)),
        showlegend=False,
    ))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF", xaxis=dict(visible=False), yaxis=dict(visible=False), dragmode="pan")
    return fig


def synthetic_demo_ledger(accounts: list[str], seed: int = 7) -> pd.DataFrame:
    """Clearly-labelled SYNTHETIC ledger for demonstrating path/cycle analysis.
    It is generated, not observed data."""
    rng = np.random.default_rng(seed)
    acc = list(rng.choice(sorted(accounts), size=min(36, len(accounts)), replace=False))
    rows = []
    start = pd.Timestamp("2025-02-01")
    for _ in range(90):
        # background transfers only flow "forward" in a random order, so they form no cycles
        i, j = sorted(rng.choice(len(acc), 2, replace=False))
        rows.append((acc[i], acc[j], round(float(rng.lognormal(7, 0.6)), 2),
                     start + pd.Timedelta(hours=int(rng.integers(0, 1400)))))
    # planted circular loop
    loop = acc[:4]
    for i in range(4):
        rows.append((loop[i], loop[(i + 1) % 4], 9500 - i * 150, start + pd.Timedelta(hours=200 + i * 3)))
    # planted layering chain
    chain = acc[4:9]
    for i in range(4):
        rows.append((chain[i], chain[i + 1], 15000 - i * 400, start + pd.Timedelta(hours=500 + i * 2)))
    df = pd.DataFrame(rows, columns=["sender_account", "receiver_account", "amount", "timestamp"])
    df["note"] = "SYNTHETIC DEMO DATA"
    return df
