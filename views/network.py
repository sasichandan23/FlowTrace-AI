import streamlit as st

from core import auth, ui
from core import graph as G
from core.model import account_summary
from core.state import get_pipeline, scope_banks, scope_label, scoped

emp = auth.require("network")
pipe = get_pipeline()
df = scoped(pipe.scored, emp)


@st.cache_resource(show_spinner="Building entity network…")
def entity_graph():
    return G.build_entity_graph(get_pipeline().scored)


EG = entity_graph()
visible = scope_banks(emp)

ui.page_header("Network analysis", "Account associations with customers, devices and merchants", scope_label(emp))

c1, c2, c3, c4 = st.columns([1, 1.6, 1, 1])
kind = c1.selectbox("Entity type", ["account", "customer", "device", "merchant"],
                    format_func=lambda k: G.NODE_STYLE[k]["label"])
col = G.COL[kind]
risk_by = df.groupby(col)["risk_score"].max().sort_values(ascending=False)
options = list(risk_by.index)
default = st.session_state.pop("network_focus", None)
idx = options.index(default) if default in options else 0
entity = c2.selectbox(f"{G.NODE_STYLE[kind]['label']} (sorted by risk)", options, index=idx,
                      format_func=lambda e: f"{e}  ·  max risk {risk_by[e]:.1f}")
radius = c3.slider("Hops", 1, 3, 2, help="How far out from the selected entity to draw")
max_nodes = c4.slider("Max nodes", 20, 150, 60, step=10)

center = G.node_id(kind, entity)
H = G.local_graph(EG, center, radius, max_nodes)

left, right = st.columns([2.2, 1])
with left:
    st.plotly_chart(G.entity_figure(H, center, visible), width="stretch",
                    config={"scrollZoom": True, "displaylogo": False})
    st.caption("Node colour = entity type · outline = risk band (red High, amber Medium, green Low, grey hidden) · "
               "customers of other banks are masked. Scroll to zoom, drag to pan.")
with right:
    d = EG.nodes[center]
    st.markdown(f"#### {G.NODE_STYLE[kind]['label']} `{entity}`")
    st.markdown(ui.risk_badge(G.band(d["risk"])) + f" &nbsp; max score **{d['risk']:.1f}**", unsafe_allow_html=True)
    nb = G.neighbours_table(EG, center, visible)
    counts = nb["type"].value_counts().to_dict() if not nb.empty else {}
    st.markdown(f"Direct links: " + ", ".join(f"**{v}** {k.lower()}(s)" for k, v in counts.items()))
    hidden = int((nb["risk_band"] == "Hidden").sum()) if not nb.empty else 0
    if hidden:
        ui.confidential(f"{hidden} linked customer(s) belong to another bank – identities hidden.")
    st.write("")
    st.markdown(f"Nodes in view: **{H.number_of_nodes()}** · links: **{H.number_of_edges()}**")
    high_nb = int((nb["risk_band"] == "High").sum()) if not nb.empty else 0
    if high_nb:
        st.markdown(ui.badge(f"{high_nb} high-risk direct neighbour(s)", "danger"), unsafe_allow_html=True)

st.markdown("##### Direct associations")
st.dataframe(nb, hide_index=True, width="stretch",
             column_config={"entity_max_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                                format="%.1f")})

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Most-shared accounts (possible mule accounts)")
    acc = account_summary(df).sort_values(["customers", "account_risk"], ascending=False).head(15)
    st.dataframe(acc[["account_id", "customers", "devices", "merchants", "account_risk", "risk_band"]],
                 hide_index=True, width="stretch",
                 column_config={"account_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                                 format="%.1f")})
with c2:
    st.markdown("##### Most-shared devices (possible coordinated rings)")
    dev = (df.groupby("device_id").agg(customers=("customer_id", "nunique"), accounts=("account_id", "nunique"),
                                       transactions=("transaction_id", "count"), max_risk=("risk_score", "max"))
           .reset_index().sort_values(["customers", "max_risk"], ascending=False).head(15))
    st.dataframe(dev, hide_index=True, width="stretch",
                 column_config={"max_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                             format="%.1f")})
