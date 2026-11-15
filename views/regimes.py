import plotly.express as px
import streamlit as st

from weather_platform import state, ui
from weather_platform.data.cities import load_cities
from weather_platform.mining.climate_regimes import build_feature_matrix, cluster_regimes
from weather_platform.viz import maps

ui.page_header("Climate regimes", "Data mining finds groups of places whose year looks alike: for example hot deserts, monsoon coasts or cool hills.")
datasets, missing = state.require_india_data()
avail = [c for c in load_cities() if c["name"] in datasets]
ui.explainer("Each location is described by 24 numbers (its average temperature and rainfall in each month). Places with similar profiles are grouped together.",
             "Same colour = similar climate year. The table names each group by its average temperature and rainfall. Scores below tell you how clearly separated the groups are.")
X, skipped = build_feature_matrix(datasets)
if skipped:
    st.caption(f"Skipped (incomplete monthly profile): {', '.join(skipped)}")
if len(X) < 6:
    ui.callout("warn", "Not enough data", "At least 6 locations with complete data are needed for clustering.")
    st.stop()
with st.container(border=True):
    feat_label = st.radio("Features used for grouping", ["24 monthly values (temperature and rainfall for each month)", "4 summary values (temperature, seasonal range, rainfall, monsoon share)"],
                          key="ind_feat", help="Running both is a robustness check: if the groups agree, the conclusion does not depend on this choice.")
    features = "monthly" if feat_label.startswith("24") else "summary"
    auto = st.checkbox("Choose the number of groups automatically (best silhouette score from 3 groups up)", value=True, key="ind_auto",
                       help="Two groups usually just separate the coldest hills from everything else, so the automatic choice starts at three. Untick to try any number from 2.")
    k = None if auto else int(st.number_input("Number of groups (k)", 2, min(8, len(X) - 1), 3, key="ind_k"))
res = cluster_regimes(X, k=k, features=features)
c1, c2, c3 = st.columns(3)
c1.metric("Groups (k)", res["k"], help=f"Best k by silhouette from 3 up: {res['best_k_by_silhouette']}. Best k overall (including 2): {res['best_k_overall']}.")
c2.metric("Agreement with a second method (ARI)", f"{res['ward_agreement_ari']:.2f}", help="Adjusted Rand index against Ward hierarchical clustering. 1 = identical groups.")
c3.metric("Stability across restarts (ARI)", f"{res['stability_ari']:.2f}", help="How similar the groups are when the algorithm is restarted with different random seeds. 1 = identical.")
from sklearn.metrics import adjusted_rand_score
_other = cluster_regimes(X, k=res["k"], features="summary" if features == "monthly" else "monthly")
_agree = adjusted_rand_score([res["labels"][n] for n in X.index], [_other["labels"][n] for n in X.index])
ui.callout("info", "Robustness check", f"Grouping the same locations with the other feature set (same k = {res['k']}) gives groups that agree with these at ARI = {_agree:.2f} (1 = identical, 0 = no more than chance).")
s_ok = res["silhouette_chosen"]
if s_ok is not None and s_ok < 0.5:
    ui.callout("warn", "Weak group structure", f"The silhouette score for {res['k']} groups is {s_ok:.2f}. Below about 0.5 the groups are not sharply separated: Indian climate varies gradually, "
               "so read these groups as a convenient partition that follows geography, not as natural classes.")
names = {int(r.cluster): f"{r.cluster}: {r.description}" for r in res["summary"].itertuples()}
st.plotly_chart(maps.category_map(avail, dict(res["labels"]), "Climate regimes", names), width="stretch", key="ind_reg_map")
st.dataframe(res["summary"].rename(columns={"cluster": "Group", "n_locations": "Locations", "mean_temp_C": "Avg temp (°C)", "annual_rain_mm": "Avg rain (mm/yr)", "rain_range_mm": "Rain range in group (mm/yr)",
                                            "wettest_month": "Wettest month", "description": "Description", "members": "Members"}), width="stretch", hide_index=True)
left, right = st.columns(2)
left.plotly_chart(px.bar(res["silhouette_by_k"], x="k", y="silhouette", title="Silhouette score by number of groups (higher is better)", template="plotly_white",
                         color_discrete_sequence=[maps.PALETTE[0]]), width="stretch", key="ind_sil")
pc = res["pca_coords"].copy()
pc["cluster"] = [str(res["labels"][n]) for n in pc.index]
pc["name"] = pc.index
right.plotly_chart(px.scatter(pc, x="PC1", y="PC2", color="cluster", hover_name="name", template="plotly_white",
                              color_discrete_map=maps.cluster_color_map(res["labels"].values()), category_orders={"cluster": [str(k) for k in sorted(set(res["labels"].values()))]},
                              title=f"Two-dimensional view ({res['pca_explained'][0]:.0%} + {res['pca_explained'][1]:.0%} of the variation)"), width="stretch", key="ind_pca")
if auto and res["best_k_overall"] == 2:
    ui.callout("info", "Why not the best silhouette?", "The silhouette score is highest for 2 groups, but that split mostly separates the coldest hill stations from everything else. "
               "The automatic choice therefore starts at three groups; all scores are shown in the chart so you can judge for yourself.")
elif not auto and res["k"] != res["best_k_overall"]:
    _s = res["silhouette_by_k"]
    _best = float(_s.loc[_s["k"] == res["best_k_overall"], "silhouette"].iloc[0])
    ui.callout("info", "Your choice of k", f"You chose {res['k']} groups. The silhouette score is highest for {res['best_k_overall']} groups ({_best:.2f}), but scores for nearby k are close, "
               "so prefer the number of groups that gives the most interpretable and stable result.")
with st.expander("Method and limitations"):
    st.write("Features: average temperature and log-transformed total rainfall of each calendar month (24 values per location), standardised. KMeans with k chosen by silhouette score, checked against Ward hierarchical clustering and random restarts.")
    for c in res["caveats"]:
        st.write("- " + c)
