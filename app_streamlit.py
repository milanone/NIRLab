"""
NIR Spectrum Manager - prototipo esplorativo (Streamlit).

Riproduce il workflow sviluppato manualmente per gli spettri NIR JCAMP-DX:
carica JDX -> converte in CSV -> spettri grezzi -> SNV -> confronto
duplicati -> PCA -> clustering di Ward -> PLS. Pensato come base da evolvere
(prima qui, poi eventualmente come app desktop).

Codice e dati sono separati: ogni analisi e' un "progetto" con la sua
cartella in data/<progetto>/ (raw/, una sottocartella per gruppo, ed
eventuali file/report comuni) — vedi nir_core.py per i dettagli.

Avvio:
    streamlit run app_streamlit.py
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

import nir_core as core

st.set_page_config(page_title="NIR Spectrum Manager", layout="wide")
st.title("NIR Spectrum Manager")
st.caption("Prototipo esplorativo — scegli o crea un progetto, poi carica uno o più file JDX.")

# ---------------------------------------------------------------------------
# 0. Progetto: ogni analisi ha la sua cartella in data/<progetto>/, cosi'
#    dataset diversi (strumenti, campagne di misura diverse) non si mescolano.
# ---------------------------------------------------------------------------

st.sidebar.header("Progetto")
existing_projects = core.list_projects()
project_choice = st.sidebar.selectbox("Progetto", existing_projects + ["+ nuovo progetto..."])

if project_choice == "+ nuovo progetto...":
    project = st.sidebar.text_input("Nome del nuovo progetto", placeholder="es. salicornia-critmo").strip()
else:
    project = project_choice

if not project:
    st.info("Crea o seleziona un progetto nella sidebar per iniziare.")
    st.stop()

core.project_dir(project)
st.sidebar.caption(f"Cartella: `data/{project}/`")

# ---------------------------------------------------------------------------
# 1. Input: file gia' presenti in data/<progetto>/raw/, o nuovi upload
# ---------------------------------------------------------------------------

raw_files_on_disk = core.list_raw_files(project)
picked_existing = st.multiselect(
    f"File JDX in data/{project}/raw/",
    options=[p.name for p in raw_files_on_disk],
    default=[p.name for p in raw_files_on_disk],
)
uploaded_files = st.file_uploader("...oppure carica nuovi file JDX", type=["jdx"], accept_multiple_files=True)

all_spectra = []
n_sources = 0
for name in picked_existing:
    text = (core.raw_dir(project) / name).read_text(encoding="utf-8", errors="replace")
    all_spectra.extend(core.parse_jdx(text))
    n_sources += 1

newly_uploaded = {}
for f in uploaded_files or []:
    text = f.read().decode("utf-8", errors="replace")
    all_spectra.extend(core.parse_jdx(text))
    newly_uploaded[f.name] = text
    n_sources += 1

if n_sources == 0:
    st.info(f"Seleziona un file già in data/{project}/raw/ oppure caricane uno nuovo per iniziare.")
    st.stop()

if newly_uploaded and st.button(f"Salva {len(newly_uploaded)} file caricati in data/{project}/raw/"):
    for name, text in newly_uploaded.items():
        (core.raw_dir(project) / name).write_text(text, encoding="utf-8")
    st.success(f"Salvati in data/{project}/raw/ — riapri la pagina per vederli nell'elenco sopra.")

st.success(f"{len(all_spectra)} spettri trovati in {n_sources} file.")

sample_names = [s.sample_name for s in all_spectra]
groups = core.detect_groups(sample_names)

# ---------------------------------------------------------------------------
# 2. Selezione gruppo di lavoro
# ---------------------------------------------------------------------------

st.sidebar.header("Gruppo di lavoro")
group_options = ["(tutti)"] + sorted(groups.keys())
chosen_group = st.sidebar.selectbox("Prefisso campione", group_options)

if chosen_group == "(tutti)":
    working_spectra = all_spectra
else:
    names_in_group = set(groups[chosen_group])
    working_spectra = [s for s in all_spectra if s.sample_name in names_in_group]

raw_df = core.spectra_to_dataframe(working_spectra)
sample_cols_all = [c for c in raw_df.columns if c != "wavelength_nm"]
lab_df = core.lab_dataframe(working_spectra)

st.sidebar.write(f"{len(sample_cols_all)} campioni nel gruppo selezionato")

st.sidebar.header("Salvataggio dati")
default_folder = chosen_group.capitalize() if chosen_group != "(tutti)" else "output"
output_folder = st.sidebar.text_input(
    f"Cartella di output (dentro data/{project}/)", value=default_folder,
    help="I risultati salvati da questa pagina finiscono in data/<progetto>/<nome cartella>/.",
)


def save_csv_button(label: str, df: pd.DataFrame, filename: str, key: str):
    if st.button(label, key=key):
        path = core.group_dir(project, output_folder) / filename
        df.to_csv(path, index=False)
        st.success(f"Salvato: {path.relative_to(core.APP_DIR)}")

# ---------------------------------------------------------------------------
# 3. Esclusioni manuali (outlier) e media duplicati
# ---------------------------------------------------------------------------

st.sidebar.header("Preprocessing")
excluded = st.sidebar.multiselect(
    "Escludi campioni (outlier)", options=sample_cols_all,
    help="Un campione va escluso per un motivo noto (es. biologico), non solo perche' 'diverso' dagli altri.",
)
do_average = st.sidebar.checkbox("Media i duplicati originale/bis", value=True)

included_cols = [c for c in sample_cols_all if c not in excluded]
clean_df = raw_df[["wavelength_nm"] + included_cols]

pairs = core.find_duplicate_pairs(included_cols)
if do_average and pairs:
    work_df = core.average_duplicates(clean_df, included_cols, pairs)
else:
    work_df = clean_df
work_cols = [c for c in work_df.columns if c != "wavelength_nm"]

lab_avg = core.average_lab_duplicates(lab_df, pairs) if do_average else lab_df

snv_df = core.compute_snv(work_df, work_cols) if len(work_cols) > 0 else work_df

tab_raw, tab_lab, tab_outlier, tab_dup, tab_snv, tab_pca, tab_cluster, tab_pls = st.tabs(
    ["Grezzi", "Dati L,a,b", "Outlier", "Duplicati", "SNV", "PCA", "Clustering", "PLS/Regressione"]
)

# ---------------------------------------------------------------------------
# Tab: Grezzi
# ---------------------------------------------------------------------------

with tab_raw:
    st.subheader("Spettri grezzi (gruppo completo, prima di esclusioni)")
    fig = go.Figure()
    for c in sample_cols_all:
        fig.add_trace(go.Scatter(x=raw_df["wavelength_nm"], y=raw_df[c], mode="lines", name=c, line=dict(width=1)))
    fig.update_layout(xaxis_title="Lunghezza d'onda (nm)", yaxis_title="Assorbanza", height=550)
    st.plotly_chart(fig, use_container_width=True)

    csv_buf = io.StringIO()
    raw_df.to_csv(csv_buf, index=False)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Scarica CSV (grezzi, gruppo completo)", csv_buf.getvalue(),
                            file_name=f"{chosen_group}_grezzi.csv", mime="text/csv")
    with col2:
        save_csv_button(f"Salva in data/{project}/{output_folder}/", raw_df, f"{output_folder}_grezzi.csv", "save_raw")

# ---------------------------------------------------------------------------
# Tab: Dati L,a,b
# ---------------------------------------------------------------------------

with tab_lab:
    st.subheader("Valori colorimetrici L, a, b (ordinati per campione)")
    if lab_df.empty:
        st.info("Nessun valore L,a,b nei blocchi JDX di questo gruppo.")
    else:
        st.dataframe(lab_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        lab_df.to_csv(csv_buf, index=False)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Scarica CSV (L,a,b)", csv_buf.getvalue(),
                                file_name=f"{chosen_group}_LAB_values.csv", mime="text/csv")
        with col2:
            if st.button(f"Salva in data/{project}/LAB_values.csv", key="save_lab"):
                path = core.project_dir(project) / "LAB_values.csv"
                lab_df.to_csv(path, index=False)
                st.success(f"Salvato: {path.relative_to(core.APP_DIR)}")

# ---------------------------------------------------------------------------
# Tab: Outlier
# ---------------------------------------------------------------------------

with tab_outlier:
    st.subheader("Distanza dalla media (SNV) — aiuto per individuare outlier")
    if len(sample_cols_all) < 2:
        st.warning("Servono almeno 2 campioni.")
    else:
        snv_full = core.compute_snv(raw_df, sample_cols_all)
        scores = core.outlier_scores(snv_full, sample_cols_all)
        st.dataframe(scores, use_container_width=True)
        st.caption(
            "Un RMSE molto più alto degli altri suggerisce un possibile outlier, ma la decisione "
            "di escluderlo va motivata (es. causa biologica nota), non presa solo sul numero. "
            "Usa la casella 'Escludi campioni' nella sidebar."
        )

# ---------------------------------------------------------------------------
# Tab: Duplicati
# ---------------------------------------------------------------------------

with tab_dup:
    st.subheader("Riproducibilità dei duplicati (originale vs 'bis')")
    if not pairs:
        st.info("Nessuna coppia originale/bis rilevata in questo gruppo (dopo le esclusioni).")
    else:
        metrics = core.duplicate_metrics(clean_df, pairs)
        st.dataframe(metrics, use_container_width=True)

        pick = st.selectbox("Coppia da visualizzare", [f"{o}/{b}" for o, b in pairs])
        orig, bis = pick.split("/")
        snv_clean = core.compute_snv(clean_df, included_cols)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snv_clean["wavelength_nm"], y=snv_clean[orig], name=orig))
            fig.add_trace(go.Scatter(x=snv_clean["wavelength_nm"], y=snv_clean[bis], name=bis))
            fig.update_layout(title=f"{orig} vs {bis} (SNV)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            diff = snv_clean[orig] - snv_clean[bis]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snv_clean["wavelength_nm"], y=diff, name="differenza"))
            fig.add_hline(y=0, line_color="gray")
            fig.update_layout(title="Spettro differenza (SNV)", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: SNV / dataset finale
# ---------------------------------------------------------------------------

with tab_snv:
    st.subheader("Dataset finale (dopo esclusioni + media duplicati) e SNV")
    st.write(f"{len(work_cols)} campioni nel dataset di lavoro: {work_cols}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for c in work_cols:
            fig.add_trace(go.Scatter(x=work_df["wavelength_nm"], y=work_df[c], mode="lines", name=c, line=dict(width=1)))
        fig.update_layout(title="Assorbanza grezza (dataset finale)", height=500,
                           xaxis_title="nm", yaxis_title="Assorbanza")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        for c in work_cols:
            fig.add_trace(go.Scatter(x=snv_df["wavelength_nm"], y=snv_df[c], mode="lines", name=c, line=dict(width=1)))
        fig.update_layout(title="SNV (dataset finale)", height=500,
                           xaxis_title="nm", yaxis_title="Assorbanza (SNV)")
        st.plotly_chart(fig, use_container_width=True)

    csv_buf = io.StringIO()
    snv_df.to_csv(csv_buf, index=False)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Scarica CSV (SNV, dataset finale)", csv_buf.getvalue(),
                            file_name=f"{chosen_group}_finale_SNV.csv", mime="text/csv")
    with col2:
        save_csv_button(f"Salva in data/{project}/{output_folder}/", snv_df, f"{output_folder}_finale_SNV.csv", "save_snv")

# ---------------------------------------------------------------------------
# Tab: PCA
# ---------------------------------------------------------------------------

with tab_pca:
    st.subheader("PCA sugli spettri SNV")
    if len(work_cols) < 3:
        st.warning("Servono almeno 3 campioni per una PCA sensata.")
    else:
        scores_df, loadings, explained = core.run_pca(snv_df, work_cols)
        merged = scores_df.copy()
        if not lab_avg.empty:
            merged = merged.join(lab_avg.set_index("sample"), how="left")

        color_options = ["(nessuno)"] + [c for c in ["L", "a", "b"] if c in merged.columns]
        color_by = st.selectbox("Colora punti per", color_options)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                merged, x="PC1", y="PC2", text=merged.index,
                color=None if color_by == "(nessuno)" else color_by,
                color_continuous_scale="RdYlGn_r",
                title=f"PCA scores — PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(explained) + 1)), y=explained, mode="lines+markers"))
            fig.update_layout(title="Scree plot", xaxis_title="Componente", yaxis_title="Varianza spiegata (%)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Loadings**")
        pc_to_show = st.selectbox("Componente", loadings.columns.tolist())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=loadings.index, y=loadings[pc_to_show], mode="lines"))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(xaxis_title="Lunghezza d'onda (nm)", yaxis_title=f"Loading {pc_to_show}", height=350)
        st.plotly_chart(fig, use_container_width=True)

        if not lab_avg.empty:
            st.markdown("**Correlazione (Pearson r) tra PC e L,a,b**")
            corr_cols = [c for c in ["L", "a", "b"] if c in merged.columns]
            corr = merged[list(scores_df.columns) + corr_cols].corr().loc[list(scores_df.columns), corr_cols]
            st.dataframe(corr.round(3), use_container_width=True)

        st.dataframe(merged.round(4), use_container_width=True)
        save_csv_button(f"Salva scores PCA in data/{project}/{output_folder}/", merged.reset_index(names="campione"),
                         f"{output_folder}_PCA_scores_lab.csv", "save_pca")

# ---------------------------------------------------------------------------
# Tab: Clustering
# ---------------------------------------------------------------------------

with tab_cluster:
    st.subheader("Clustering gerarchico di Ward")
    if len(work_cols) < 4:
        st.warning("Servono almeno 4 campioni per scegliere automaticamente k.")
    else:
        Z, sil_scores, best_k, assignment = core.run_ward_clustering(snv_df, work_cols)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            dendrogram(Z, labels=work_cols, ax=ax)
            ax.set_title(f"Dendrogramma Ward (k scelto = {best_k})")
            ax.set_ylabel("Distanza")
            st.pyplot(fig)
        with col2:
            sil_df = pd.DataFrame({"k": list(sil_scores.keys()), "silhouette": list(sil_scores.values())})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sil_df["k"], y=sil_df["silhouette"], mode="lines+markers"))
            fig.add_vline(x=best_k, line_dash="dash", line_color="red")
            fig.update_layout(title="Selezione automatica di k (silhouette)",
                               xaxis_title="k", yaxis_title="Silhouette score")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**k scelto automaticamente: {best_k}** (silhouette = {sil_scores[best_k]:.3f})")
        assignment_sorted = assignment.sort_values()
        st.dataframe(assignment_sorted, use_container_width=True)
        save_csv_button(f"Salva assegnazioni in data/{project}/{output_folder}/",
                         assignment_sorted.rename_axis("campione").reset_index(),
                         f"{output_folder}_Ward_clusters.csv", "save_ward")

with tab_pls:
    st.subheader("PLS — regressione degli spettri SNV su una variabile target")

    if lab_avg.empty:
        st.warning("Nessun valore L,a,b trovato per questo gruppo: non c'e' una variabile target da predire.")
    else:
        target_options = [c for c in ["L", "a", "b"] if c in lab_avg.columns]
        target = st.selectbox("Variabile target", target_options)

        lab_indexed = lab_avg.set_index("sample")
        common = [c for c in work_cols if c in lab_indexed.index and pd.notna(lab_indexed.loc[c, target])]
        missing = [c for c in work_cols if c not in common]
        if missing:
            st.caption(f"Campioni esclusi per L,a,b mancante: {missing}")

        if len(common) < 6:
            st.warning("Servono almeno ~6 campioni con target noto per una PLS sensata.")
        else:
            X = snv_df[common].values.T
            y = lab_indexed.loc[common, target].values.astype(float)

            max_comp = st.slider("Numero massimo di componenti da esplorare", 2, min(15, len(common) - 2), min(8, len(common) - 2))
            rmsecv, best_n = core.pls_cross_validate(X, y, max_components=max_comp)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(rmsecv.keys()), y=list(rmsecv.values()), mode="lines+markers"))
            fig.add_vline(x=best_n, line_dash="dash", line_color="red")
            fig.update_layout(title="RMSECV (leave-one-out) al variare del numero di componenti",
                               xaxis_title="Numero di componenti PLS", yaxis_title=f"RMSECV ({target})", height=350)
            st.plotly_chart(fig, use_container_width=True)

            n_components = st.number_input(
                "Componenti PLS da usare per il modello finale",
                min_value=1, max_value=max_comp, value=best_n,
                help="Precompilato con il valore che minimizza l'RMSECV, ma puoi cambiarlo (es. per un modello piu' parsimonioso).",
            )

            pls, y_pred_cal, y_pred_cv, metrics = core.fit_pls(X, y, n_components)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² calibrazione", f"{metrics['r2_cal']:.3f}")
            col2.metric("R² CV (LOO)", f"{metrics['r2_cv']:.3f}")
            col3.metric("RMSE calibrazione", f"{metrics['rmse_cal']:.3f}")
            col4.metric("RMSE CV (LOO)", f"{metrics['rmse_cv']:.3f}")
            st.caption(
                "Con pochi campioni il valore di calibrazione è ottimistico: guarda soprattutto R²/RMSE "
                "in cross-validazione (LOO) per valutare la capacità predittiva reale."
            )

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y, y=y_pred_cal, mode="markers+text", text=common,
                                          textposition="top center", name="calibrazione"))
                fig.add_trace(go.Scatter(x=y, y=y_pred_cv, mode="markers", name="LOO-CV", marker=dict(symbol="x")))
                lims = [min(y.min(), y_pred_cal.min(), y_pred_cv.min()), max(y.max(), y_pred_cal.max(), y_pred_cv.max())]
                fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dot", color="gray"), name="ideale"))
                fig.update_layout(title=f"Predetto vs osservato ({target})",
                                   xaxis_title=f"{target} osservato", yaxis_title=f"{target} predetto", height=450)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                coef = pls.coef_.ravel()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=snv_df["wavelength_nm"], y=coef, mode="lines"))
                fig.add_hline(y=0, line_color="gray")
                fig.update_layout(title=f"Coefficienti di regressione PLS ({n_components} componenti)",
                                   xaxis_title="Lunghezza d'onda (nm)", yaxis_title="Coefficiente", height=450)
                st.plotly_chart(fig, use_container_width=True)

            results_df = pd.DataFrame({
                "campione": common, f"{target}_osservato": y,
                f"{target}_predetto_cal": y_pred_cal, f"{target}_predetto_cv": y_pred_cv,
            })
            st.dataframe(results_df.round(4), use_container_width=True)
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Scarica risultati PLS (CSV)", csv_buf.getvalue(),
                                    file_name=f"{chosen_group}_PLS_{target}.csv", mime="text/csv")
            with col2:
                save_csv_button(f"Salva in data/{project}/{output_folder}/", results_df,
                                 f"{output_folder}_PLS_{target}.csv", "save_pls")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Passi successivi possibili: confronto diretto multi-gruppo, esportazione report."
)
