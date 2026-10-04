"""
Streamlit frontend: chat with the papers (or your own PDFs), and the two
benchmark dashboards.

Run (from the repo root, with the API running):
  streamlit run src/ui/app.py
  # API_URL=http://localhost:8000 by default

Pages
  Ask the papers          chat with follow-up questions; upload PDFs and choose
                          whether the paper corpus is searched too; every answer
                          shows its sources, rebuilt tables and figures
  Uncertainty benchmark   CIFAR-10 vs SVHN / CIFAR-100: OOD detection,
                          calibration, corruption, selective prediction
  Retrieval evaluation    Recall/MRR/NDCG per retrieval configuration, with
                          significance and abstention, plus generation quality
  About                   what this is and how it is built
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import client  # noqa: E402
from ui.client import ApiError  # noqa: E402

REPO_URL = "https://github.com/bhatsajid18/uncertainty-rag"
METHOD_LABELS = {"softmax": "Softmax (MSP)", "ensemble": "Deep Ensemble (5)",
                 "mcdropout": "MC Dropout", "edl": "Evidential (EDL)"}

st.set_page_config(page_title="Research Intelligence", page_icon="📄", layout="wide")


# --- cached API reads --------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def _health():
    return client.health()


@st.cache_data(ttl=3600, show_spinner=False)
def _figure(name: str) -> bytes:
    return client.figure(name)


@st.cache_data(ttl=60, show_spinner=False)
def _uncertainty(fake: bool):
    return client.uncertainty(fake)


@st.cache_data(ttl=3600, show_spinner=False)
def _uncertainty_figure(name: str, fake: bool) -> bytes:
    return client.uncertainty_figure(name, fake)


@st.cache_data(ttl=60, show_spinner=False)
def _eval_runs(kind: str):
    return client.eval_runs(kind)


@st.cache_data(ttl=60, show_spinner=False)
def _eval_run(kind: str, run: str):
    return client.eval_run(kind, run)


def api_status():
    try:
        h = _health()
        if h.get("retriever_loaded"):
            loaded = f"{h['papers']} papers, {h['corpus_chunks']} chunks"
        elif h.get("loading"):
            loaded = "loading the models (the first start downloads ~1.5 GB)"
        else:
            loaded = "models load on first question"
        st.sidebar.success(f"API connected - {loaded}")
        if h.get("llm_ready") is False:
            key = "GEMINI_API_KEY" if h.get("provider") == "gemini" else "GROQ_API_KEY"
            st.sidebar.warning(f"No LLM key: add {key} to .env and restart the API. "
                               "Until then questions return an error; the dashboards "
                               "work.")
        return h
    except ApiError as e:
        st.sidebar.error(str(e))
        return None


# --- chat --------------------------------------------------------------------------------


def _session() -> dict:
    if "session" not in st.session_state:
        st.session_state.session = client.new_session()
    return st.session_state.session


def _in_session(call, *args):
    """Call the API for this conversation. If the server no longer knows it (it
    expired, or the API restarted), start a new one instead of failing."""
    try:
        return call(*args)
    except ApiError as e:
        if e.status == 404:
            st.session_state.pop("session", None)
            st.warning("The conversation expired or the API restarted, so a new one "
                       "was started. Please try again.")
        else:
            st.error(str(e))
        return None


def render_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            year = f" ({s['year']})" if s.get("year") else ""
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            st.markdown(f"**[S{s['n']}]** {title}{year}, {s['pages']}  "
                        f"<span style='color:grey'>score {s['score']:.3f}</span>",
                        unsafe_allow_html=True)
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
            if s.get("table_markdown"):
                how = ("rebuilt from the PDF's column positions"
                       if s.get("table_source") == "geometry" else "rebuilt by the LLM")
                st.caption(f"Table ({how}):")
                for grid in s["table_markdown"].split("\n\n"):
                    st.markdown(grid)
            for f in s.get("figures", []):
                try:
                    st.image(_figure(f["name"]), caption=f["name"], width=420)
                except ApiError:
                    st.caption(f"(figure {f['name']} not available)")
            st.divider()


def render_turn(turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["standalone"] != turn["question"]:
            st.caption(f"Searched for: {turn['standalone']}")
        st.markdown(turn["answer"])
        if turn["sources"]:
            render_sources(turn["sources"])


def page_chat():
    st.title("Ask the papers")
    try:
        session = _session()
    except ApiError as e:
        st.error(f"Could not start a conversation: {e}")
        return

    with st.sidebar:
        st.subheader("Your PDFs")
        files = st.file_uploader("Upload papers (PDF, up to 5)", type=["pdf"],
                                 accept_multiple_files=True)
        with_corpus = st.toggle("Also search the paper corpus", value=True)
        if st.button("Use these PDFs", disabled=not files, type="primary"):
            with st.spinner("Reading and indexing your PDFs ..."):
                updated = _in_session(client.upload, session["session_id"],
                                      [(f.name, f.getvalue()) for f in files], with_corpus)
            if updated:
                st.session_state.session = updated
                st.rerun()
        if session.get("documents"):
            for d in session["documents"]:
                st.caption(f"📄 {d['title'][:60]} ({d['chunks']} chunks)")
            if st.button("Back to the paper corpus"):
                updated = _in_session(client.clear_documents, session["session_id"])
                if updated:
                    st.session_state.session = updated
                    st.rerun()
        st.divider()
        if st.button("New conversation"):
            st.session_state.pop("session", None)
            st.rerun()

    scope = {"corpus": "the paper corpus", "uploads": "your PDFs only",
             "uploads+corpus": "your PDFs and the paper corpus"}[session["scope"]]
    st.caption(f"Answers come from {scope}, with a citation for every claim. "
               "Follow-up questions work; ask about methods, reported numbers "
               "or comparisons across papers.")

    for turn in session.get("turns", []):
        render_turn(turn)

    question = st.chat_input("e.g. What error does EnD2 get on CIFAR-10?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and answering ..."):
                turn = _in_session(client.ask, session["session_id"], question)
            if turn is None:
                return
        session.setdefault("turns", []).append(turn)
        st.rerun()


# --- uncertainty dashboard ------------------------------------------------------------------


def _pct(v):
    return None if v is None else round(100 * v, 1)


def page_uncertainty():
    st.title("Uncertainty benchmark")
    st.caption("ResNet-18 trained on CIFAR-10. Can each method tell when an input is "
               "unlike its training data (SVHN: far OOD; CIFAR-100: near OOD), stay "
               "calibrated, and flag its own mistakes?")
    fake = st.toggle("Show the smoke-test results (fake data)", value=False)
    try:
        res = _uncertainty(fake)
    except ApiError as e:
        if e.status == 404:
            st.info("No results yet. Run `notebooks/kaggle_uncertainty.ipynb` on a "
                    "free Kaggle GPU, download `uncertainty_results.zip` and unzip it "
                    "in the repo root. Or run the fake-data smoke test: "
                    "`make uncertainty-smoke`.")
        else:
            st.error(str(e))
        return
    if res.get("fake"):
        st.warning("These come from the fake-data smoke test: they show the pipeline "
                   "works, and the numbers mean nothing.")

    head = pd.DataFrame([{
        "Method": METHOD_LABELS.get(r["method"], r["method"]), "Score": r["score"],
        "Accuracy %": _pct(r["accuracy"]), "NLL": r["nll"], "ECE": r["ece"],
        "Brier": r["brier"], "SVHN AUROC %": _pct(r.get("svhn_auroc")),
        "SVHN FPR95 %": _pct(r.get("svhn_fpr95")),
        "C100 AUROC %": _pct(r.get("cifar100_auroc")),
        "C100 FPR95 %": _pct(r.get("cifar100_fpr95")), "AURC": r["aurc"],
    } for r in res["headline"]])
    st.subheader("Headline")
    st.dataframe(head, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("NLL", "ECE", "Brier", "AURC")})
    st.caption("OOD detection scores each method by its own uncertainty: max softmax "
               "probability, mutual information (ensemble, MC Dropout) or vacuity (EDL). "
               "AURC ranks every method by confidence. Lower is better for NLL, ECE, "
               "Brier, FPR95 and AURC.")

    left, right = st.columns(2)
    with left:
        st.subheader("Near vs far OOD")
        nf = pd.DataFrame([{"Method": METHOD_LABELS.get(r["method"], r["method"]),
                            "OOD set": kind, "AUROC": v}
                           for r in res["near_far"]
                           for kind, v in (("Far (SVHN)", r["far_auroc"]),
                                           ("Near (CIFAR-100)", r["near_auroc"]))])
        if not nf.empty:
            # horizontal bars from zero: a bar chart with a cut axis exaggerates gaps
            st.altair_chart(alt.Chart(nf).mark_bar().encode(
                y=alt.Y("Method:N", title=None, sort=None),
                x=alt.X("AUROC:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("OOD set:N", legend=alt.Legend(orient="bottom")),
                yOffset="OOD set:N", tooltip=["Method", "OOD set", alt.Tooltip(
                    "AUROC:Q", format=".3f")]), width="stretch")
    with right:
        st.subheader("Selective prediction")
        sel = pd.DataFrame([{"Method": METHOD_LABELS.get(r["method"], r["method"]),
                             "AURC": r["aurc"], "Acc @ 80% %": _pct(r["acc_at_80"]),
                             "Acc @ 90% %": _pct(r["acc_at_90"]),
                             "Acc @ 100% %": _pct(r["acc_at_100"])}
                            for r in res["selective"]])
        st.dataframe(sel, hide_index=True, width="stretch")
        st.caption("Answer only the inputs the model is most confident about (1 - max "
                   "probability of the averaged prediction, for every method): accuracy "
                   "should rise as coverage falls if confidence flags the mistakes.")

    sev = pd.DataFrame(res.get("corruption_by_severity", []))
    if not sev.empty:
        st.subheader("Under corruption")
        sev["Method"] = sev["method"].map(lambda m: METHOD_LABELS.get(m, m))
        cols = st.columns(3)
        for col, (key, title) in zip(cols, (("accuracy", "Accuracy"), ("ece", "ECE"),
                                            ("shift_auroc", "Shift AUROC"))):
            with col:
                st.altair_chart(alt.Chart(sev).mark_line(point=True).encode(
                    x=alt.X("severity:O", title="Severity"),
                    y=alt.Y(f"{key}:Q", title=title), color="Method:N",
                    tooltip=["Method", "severity", alt.Tooltip(f"{key}:Q", format=".3f")]),
                    width="stretch")
        st.caption("Mean over 5 corruption types (noise, blur, contrast, brightness, "
                   "pixelation). Shift AUROC: how well the uncertainty separates "
                   "corrupted from clean inputs.")

    figures = [("roc_svhn.png", "ROC: SVHN"), ("roc_cifar100.png", "ROC: CIFAR-100"),
               ("reliability.png", "Reliability"), ("risk_coverage.png", "Risk-coverage")]
    tabs = st.tabs([t for _, t in figures])
    for tab, (name, _) in zip(tabs, figures):
        with tab:
            try:
                st.image(_uncertainty_figure(name, fake))
            except ApiError:
                st.caption("Figure not generated.")


# --- retrieval evaluation ------------------------------------------------------------------


def page_retrieval():
    st.title("Retrieval evaluation")
    st.caption("87 labelled questions (67 answerable, 20 unanswerable) in four "
               "categories, scored for every retrieval configuration.")
    try:
        runs = _eval_runs("retrieval")
    except ApiError as e:
        st.error(str(e))
        return
    if not runs:
        st.info("No evaluation runs yet: python src/evaluation/run_eval.py --suite core")
        return
    default = next((i for i, r in enumerate(runs) if r.startswith("core")), 0)
    run = st.selectbox("Run", runs, index=default)
    try:
        data = _eval_run("retrieval", run)
    except ApiError as e:
        st.error(str(e))
        return

    summary = pd.DataFrame(data.get("summary", []))
    if not summary.empty:
        keep = [c for c in ("config", "recall@1", "recall@3", "recall@5", "recall@10",
                            "ndcg@5", "mrr") if c in summary]
        st.subheader("Headline metrics")
        st.dataframe(summary[keep], hide_index=True, width="stretch")
        melted = summary.melt(id_vars="config", value_vars=[c for c in ("recall@5", "mrr")
                                                            if c in summary],
                              var_name="metric")
        # horizontal, so 5 or 18 configuration names all stay readable
        st.altair_chart(alt.Chart(melted, height=max(160, 44 * len(summary))).mark_bar()
                        .encode(y=alt.Y("config:N", title=None, sort=None,
                                        axis=alt.Axis(labelLimit=260)),
                                x=alt.X("value:Q", title=None,
                                        scale=alt.Scale(domain=[0, 1]),
                                        axis=alt.Axis(tickCount=5, format=".1f")),
                                color=alt.Color("metric:N",
                                                legend=alt.Legend(orient="bottom")),
                                yOffset="metric:N",
                                tooltip=["config", "metric",
                                         alt.Tooltip("value:Q", format=".3f")]),
                        width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        if data.get("significance"):
            st.subheader("Against dense-only (95% CI)")
            st.dataframe(pd.DataFrame(data["significance"]), hide_index=True,
                         width="stretch")
    with c2:
        if data.get("abstention"):
            st.subheader("Abstention (unanswerable questions)")
            st.dataframe(pd.DataFrame(data["abstention"]), hide_index=True,
                         width="stretch")

    by_cat = pd.DataFrame(data.get("summary_by_category", []))
    if not by_cat.empty and {"config", "category", "recall@5"} <= set(by_cat):
        st.subheader("Recall@5 by question type")
        st.dataframe(by_cat.pivot_table(index="config", columns="category",
                                        values="recall@5").round(3), width="stretch")

    try:
        gen_runs = _eval_runs("generation")
    except ApiError:
        gen_runs = []
    if gen_runs:
        st.subheader("Answer quality (generation evaluation)")
        try:
            g = _eval_run("generation", st.selectbox("Generation run", gen_runs))
        except ApiError as e:
            st.error(str(e))
            return
        info = g.get("run_info") or {}
        if info.get("complete") is False:
            st.warning(f"Incomplete run: {info.get('pairs_done')} of "
                       f"{info.get('pairs_total')} answers were judged, so the "
                       f"configurations are compared on the {info.get('n_compared')} "
                       "questions all of them finished. Run `make eval-gen` again to "
                       "complete it.")
        st.dataframe(pd.DataFrame(g.get("summary", [])), hide_index=True, width="stretch")
        st.caption("Faithfulness: share of answer sentences a retrieved source supports. "
                   "Citation accuracy: cited source actually supports the sentence. "
                   "Abstention: unanswerable questions correctly refused.")


# --- about ----------------------------------------------------------------------------------


def page_about():
    st.title("Uncertainty-Aware Research Intelligence")
    st.markdown(f"""
Two systems in one app, built without an LLM framework ([source]({REPO_URL})):

**1. A research assistant over ML papers.** Hybrid retrieval (bge embeddings + FAISS,
BM25, Reciprocal Rank Fusion), a cross-encoder reranker, tables rebuilt from the PDF's
geometry, figure captions and descriptions, and grounded answers that cite paper and page
for every claim, or say the sources don't cover the question. Follow-up questions and
uploaded PDFs work.

**2. An uncertainty benchmark.** Softmax, MC Dropout, Deep Ensembles and Evidential Deep
Learning on CIFAR-10, tested on SVHN (far OOD), CIFAR-100 (near OOD) and corrupted inputs:
AUROC, AUPR, FPR@95TPR, ECE, Brier and selective prediction.

Both are measured, not demonstrated: the retrieval page shows Recall/MRR/NDCG with
bootstrap confidence intervals for every configuration.
""")


PAGES = {"Ask the papers": page_chat, "Uncertainty benchmark": page_uncertainty,
         "Retrieval evaluation": page_retrieval, "About": page_about}

st.sidebar.title("Research Intelligence")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
api_status()
PAGES[choice]()
