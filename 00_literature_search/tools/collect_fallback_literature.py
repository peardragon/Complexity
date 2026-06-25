#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

import arxiv
import requests
from ddgs import DDGS


TOPIC = (
    "local entropy, Franz-Parisi potential, shell sampling, and solution-space "
    "geometry for neural networks and dataset complexity"
)

FACETS = [
    {
        "id": "local_entropy_flat_minima",
        "query": '"local entropy" neural networks flat minima wide valleys',
        "notes": "Core local-entropy/flat-minima bridge.",
    },
    {
        "id": "franz_parisi_perceptron",
        "query": '"Franz-Parisi" potential perceptron neural network local entropy',
        "notes": "Statistical-physics lineage and perceptron reference theory.",
    },
    {
        "id": "robust_ensembles_dense_clusters",
        "query": '"robust ensemble" "dense clusters" neural networks local entropy',
        "notes": "Baldassi-Zecchina dense-cluster line.",
    },
    {
        "id": "entropy_sgd_wide_valleys",
        "query": '"Entropy-SGD" "wide valleys" local entropy deep learning',
        "notes": "Optimization methods that explicitly bias toward local entropy.",
    },
    {
        "id": "sharpness_flatness_generalization",
        "query": 'sharpness flatness generalization neural networks reparameterization PAC-Bayes',
        "notes": "Adjacent flatness and PAC-Bayes debate.",
    },
    {
        "id": "loss_landscape_connectivity",
        "query": '"loss landscape" "mode connectivity" neural networks flat minima',
        "notes": "Solution-space topology and connectivity.",
    },
    {
        "id": "random_labels_dataset_complexity",
        "query": '"random labels" memorization dataset complexity neural networks graph total variation',
        "notes": "Dataset/rule complexity and label randomization baselines.",
    },
    {
        "id": "partition_function_sampling",
        "query": '"annealed importance sampling" "sequential Monte Carlo" partition function neural networks',
        "notes": "Normalizing-constant estimation and SMC methods.",
    },
    {
        "id": "von_mises_fisher_shell",
        "query": '"von Mises-Fisher" sampling hypersphere shell importance sampling',
        "notes": "Directional shell proposal and hypersphere sampling.",
    },
    {
        "id": "recent_density_states",
        "query": '"density of states" neural networks Franz-Parisi local geometry loss landscape',
        "notes": "Recent density-of-states/local geometry papers.",
    },
]

SEED_RECORDS: list[dict[str, Any]] = [
    {
        "title": "Recipes for metastable states in spin glasses",
        "authors": ["Silvio Franz", "Giorgio Parisi"],
        "year": 1995,
        "doi": "10.1051/jp1:1995201",
        "arxiv_id": "cond-mat/9503167",
        "venue": "Journal de Physique I",
        "url": "https://arxiv.org/abs/cond-mat/9503167",
        "pdf_url": "https://arxiv.org/pdf/cond-mat/9503167",
        "abstract": "Introduces the constrained-overlap potential used to study metastable states.",
        "source": "manual_seed",
        "query": "Franz-Parisi potential seminal seed",
    },
    {
        "title": "Annealed importance sampling",
        "authors": ["Radford M. Neal"],
        "year": 2001,
        "doi": "10.1023/A:1008923215028",
        "arxiv_id": "physics/9803008",
        "venue": "Statistics and Computing",
        "url": "https://arxiv.org/abs/physics/9803008",
        "pdf_url": "https://arxiv.org/pdf/physics/9803008",
        "abstract": "Uses annealing sequences to construct an importance sampler for normalizing constants.",
        "source": "manual_seed",
        "query": "normalizing constant estimation seminal seed",
    },
    {
        "title": "Sequential Monte Carlo samplers",
        "authors": ["Pierre Del Moral", "Arnaud Doucet", "Ajay Jasra"],
        "year": 2006,
        "doi": "10.1111/j.1467-9868.2006.00553.x",
        "venue": "Journal of the Royal Statistical Society: Series B",
        "url": "https://doi.org/10.1111/j.1467-9868.2006.00553.x",
        "abstract": "Sequentially samples from distributions known up to normalizing constants.",
        "source": "manual_seed",
        "query": "SMC sampler seminal seed",
    },
    {
        "title": "Simulation of the von Mises Fisher distribution",
        "authors": ["Andrew T. A. Wood"],
        "year": 1994,
        "doi": "10.1080/03610919408813161",
        "venue": "Communications in Statistics - Simulation and Computation",
        "url": "https://doi.org/10.1080/03610919408813161",
        "abstract": "Provides a practical algorithm for sampling from von Mises-Fisher distributions.",
        "source": "manual_seed",
        "query": "vMF sampling seminal seed",
    },
    {
        "title": "Flat Minima",
        "authors": ["Sepp Hochreiter", "Jürgen Schmidhuber"],
        "year": 1997,
        "doi": "10.1162/neco.1997.9.1.1",
        "venue": "Neural Computation",
        "url": "https://doi.org/10.1162/neco.1997.9.1.1",
        "abstract": "Classic argument relating flat minima to generalization.",
        "source": "manual_seed",
        "query": "flat minima seminal seed",
    },
    {
        "title": "Subdominant Dense Clusters Allow for Simple Learning and High Computational Performance in Neural Networks with Discrete Synapses",
        "authors": ["Carlo Baldassi", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2015,
        "doi": "10.1103/PhysRevLett.115.128101",
        "arxiv_id": "1509.05753",
        "venue": "Physical Review Letters",
        "url": "https://arxiv.org/abs/1509.05753",
        "pdf_url": "https://arxiv.org/pdf/1509.05753",
        "abstract": "Shows rare dense solution clusters in discrete-synapse perceptrons.",
        "source": "manual_seed",
        "query": "dense clusters seminal seed",
    },
    {
        "title": "Local entropy as a measure for sampling solutions in Constraint Satisfaction Problems",
        "authors": ["Carlo Baldassi", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2016,
        "doi": "10.1088/1742-5468/2016/02/023301",
        "arxiv_id": "1511.05634",
        "venue": "Journal of Statistical Mechanics: Theory and Experiment",
        "url": "https://arxiv.org/abs/1511.05634",
        "pdf_url": "https://arxiv.org/pdf/1511.05634",
        "abstract": "Introduces Entropy-driven Monte Carlo for sampling solutions via local entropy.",
        "source": "manual_seed",
        "query": "local entropy CSP seminal seed",
    },
    {
        "title": "Unreasonable effectiveness of learning neural networks: From accessible states and robust ensembles to basic algorithmic schemes",
        "authors": ["Carlo Baldassi", "Christian Borgs", "Jennifer Chayes", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2016,
        "doi": "10.1073/pnas.1608103113",
        "arxiv_id": "1605.06444",
        "venue": "Proceedings of the National Academy of Sciences",
        "url": "https://arxiv.org/abs/1605.06444",
        "pdf_url": "https://arxiv.org/pdf/1605.06444",
        "abstract": "Develops robust ensembles and algorithms targeting accessible dense states.",
        "source": "manual_seed",
        "query": "robust ensemble seminal seed",
    },
    {
        "title": "Entropy-SGD: Biasing Gradient Descent Into Wide Valleys",
        "authors": ["Pratik Chaudhari", "Anna Choromanska", "Stefano Soatto", "Yann LeCun", "Carlo Baldassi", "Christian Borgs", "Jennifer Chayes", "Levent Sagun", "Riccardo Zecchina"],
        "year": 2017,
        "arxiv_id": "1611.01838",
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1611.01838",
        "pdf_url": "https://arxiv.org/pdf/1611.01838",
        "abstract": "Optimizes a local-entropy objective to favor wide valleys in neural-network loss landscapes.",
        "source": "manual_seed",
        "query": "Entropy-SGD seminal seed",
    },
    {
        "title": "Shaping the learning landscape in neural networks around wide flat minima",
        "authors": ["Carlo Baldassi", "Fabrizio Pittorino", "Riccardo Zecchina"],
        "year": 2020,
        "doi": "10.1073/pnas.1908636117",
        "arxiv_id": "1905.07833",
        "venue": "Proceedings of the National Academy of Sciences",
        "url": "https://arxiv.org/abs/1905.07833",
        "pdf_url": "https://arxiv.org/pdf/1905.07833",
        "abstract": "Studies wide flat minima in one- and two-layer neural-network models.",
        "source": "manual_seed",
        "query": "wide flat minima statistical physics seed",
    },
    {
        "title": "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima",
        "authors": ["Nitish Shirish Keskar", "Dheevatsa Mudigere", "Jorge Nocedal", "Mikhail Smelyanskiy", "Ping Tak Peter Tang"],
        "year": 2017,
        "arxiv_id": "1609.04836",
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1609.04836",
        "pdf_url": "https://arxiv.org/pdf/1609.04836",
        "abstract": "Links large-batch training to sharper minima and generalization gaps.",
        "source": "manual_seed",
        "query": "sharp minima large batch seed",
    },
    {
        "title": "Sharp Minima Can Generalize For Deep Nets",
        "authors": ["Laurent Dinh", "Razvan Pascanu", "Samy Bengio", "Yoshua Bengio"],
        "year": 2017,
        "arxiv_id": "1703.04933",
        "venue": "International Conference on Machine Learning",
        "url": "https://arxiv.org/abs/1703.04933",
        "pdf_url": "https://arxiv.org/pdf/1703.04933",
        "abstract": "Shows that common flatness measures can be manipulated by reparameterization.",
        "source": "manual_seed",
        "query": "flatness critique seed",
    },
    {
        "title": "Understanding deep learning requires rethinking generalization",
        "authors": ["Chiyuan Zhang", "Samy Bengio", "Moritz Hardt", "Benjamin Recht", "Oriol Vinyals"],
        "year": 2017,
        "arxiv_id": "1611.03530",
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1611.03530",
        "pdf_url": "https://arxiv.org/pdf/1611.03530",
        "abstract": "Demonstrates that standard networks can fit random labels, challenging simple complexity explanations.",
        "source": "manual_seed",
        "query": "random labels generalization seed",
    },
    {
        "title": "A Closer Look at Memorization in Deep Networks",
        "authors": ["Devansh Arpit", "Stanislaw Jastrzebski", "Nicolas Ballas", "David Krueger", "Emmanuel Bengio", "Maxinder S. Kanwal", "Tegan Maharaj", "Asja Fischer", "Aaron Courville", "Yoshua Bengio", "Simon Lacoste-Julien"],
        "year": 2017,
        "arxiv_id": "1706.05394",
        "venue": "International Conference on Machine Learning",
        "url": "https://arxiv.org/abs/1706.05394",
        "pdf_url": "https://arxiv.org/pdf/1706.05394",
        "abstract": "Separates learning simple patterns from memorizing random labels.",
        "source": "manual_seed",
        "query": "memorization random labels seed",
    },
    {
        "title": "Visualizing the Loss Landscape of Neural Nets",
        "authors": ["Hao Li", "Zheng Xu", "Gavin Taylor", "Christoph Studer", "Tom Goldstein"],
        "year": 2018,
        "arxiv_id": "1712.09913",
        "venue": "Advances in Neural Information Processing Systems",
        "url": "https://arxiv.org/abs/1712.09913",
        "pdf_url": "https://arxiv.org/pdf/1712.09913",
        "abstract": "Introduces visualization methods for neural-network loss landscapes.",
        "source": "manual_seed",
        "query": "loss landscape visualization seed",
    },
    {
        "title": "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs",
        "authors": ["Timur Garipov", "Pavel Izmailov", "Dmitrii Podoprikhin", "Dmitry Vetrov", "Andrew Gordon Wilson"],
        "year": 2018,
        "arxiv_id": "1802.10026",
        "venue": "Advances in Neural Information Processing Systems",
        "url": "https://arxiv.org/abs/1802.10026",
        "pdf_url": "https://arxiv.org/pdf/1802.10026",
        "abstract": "Shows low-loss curves connecting independently trained neural networks.",
        "source": "manual_seed",
        "query": "mode connectivity seed",
    },
    {
        "title": "Essentially No Barriers in Neural Network Energy Landscape",
        "authors": ["Felix Draxler", "Kambis Veschgini", "Manfred Salmhofer", "Fred A. Hamprecht"],
        "year": 2018,
        "arxiv_id": "1803.00885",
        "venue": "International Conference on Machine Learning",
        "url": "https://arxiv.org/abs/1803.00885",
        "pdf_url": "https://arxiv.org/pdf/1803.00885",
        "abstract": "Finds low-loss paths between independently trained minima.",
        "source": "manual_seed",
        "query": "mode connectivity seed",
    },
    {
        "title": "Fantastic Generalization Measures and Where to Find Them",
        "authors": ["Yiding Jiang", "Behnam Neyshabur", "Hossein Mobahi", "Dilip Krishnan", "Samy Bengio"],
        "year": 2020,
        "arxiv_id": "1912.02178",
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1912.02178",
        "pdf_url": "https://arxiv.org/pdf/1912.02178",
        "abstract": "Large-scale empirical comparison of many generalization measures.",
        "source": "manual_seed",
        "query": "generalization measures seed",
    },
    {
        "title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "authors": ["Pierre Foret", "Ariel Kleiner", "Hossein Mobahi", "Behnam Neyshabur"],
        "year": 2021,
        "arxiv_id": "2010.01412",
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/2010.01412",
        "pdf_url": "https://arxiv.org/pdf/2010.01412",
        "abstract": "Optimizes neighborhoods around parameters via sharpness-aware minimization.",
        "source": "manual_seed",
        "query": "SAM flatness seed",
    },
    {
        "title": "A Modern Look at the Relationship between Sharpness and Generalization",
        "authors": ["Maksym Andriushchenko", "Francesco Croce", "Maximilian Müller", "Matthias Hein", "Nicolas Flammarion"],
        "year": 2023,
        "doi": "10.5555/3618408.3618444",
        "arxiv_id": "2302.07011",
        "venue": "International Conference on Machine Learning",
        "url": "https://arxiv.org/abs/2302.07011",
        "pdf_url": "https://arxiv.org/pdf/2302.07011",
        "abstract": "Comprehensively tests sharpness-generalization correlations in modern settings.",
        "source": "manual_seed",
        "query": "sharpness limitation seed",
    },
]


def slugify(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:max_len].strip("-") or "item"


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "literature-radar/0.1 (mailto:unknown@example.com)"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def search_arxiv(query: str, limit: int) -> list[dict[str, Any]]:
    client = arxiv.Client(page_size=min(100, max(10, limit)), delay_seconds=1.0, num_retries=2)
    records = []
    search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
    for result in client.results(search):
        records.append(
            {
                "title": clean_text(result.title),
                "authors": [str(author) for author in result.authors],
                "year": result.published.year if result.published else None,
                "doi": result.doi,
                "arxiv_id": result.entry_id.rsplit("/", 1)[-1],
                "pdf_url": result.pdf_url,
                "url": result.entry_id,
                "venue": "arXiv",
                "abstract": clean_text(result.summary),
                "source": "arxiv",
                "query": query,
            }
        )
    return records


def search_semantic_scholar(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,venue,externalIds,openAccessPdf,citationCount,abstract,url",
    }
    data = request_json("https://api.semanticscholar.org/graph/v1/paper/search", params=params)
    records = []
    for paper in data.get("data") or []:
        ext = paper.get("externalIds") or {}
        oa_pdf = paper.get("openAccessPdf") or {}
        records.append(
            {
                "title": clean_text(paper.get("title")),
                "authors": [a.get("name") for a in (paper.get("authors") or []) if a.get("name")],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "pdf_url": oa_pdf.get("url"),
                "url": paper.get("url"),
                "citations": paper.get("citationCount"),
                "abstract": clean_text(paper.get("abstract")),
                "source": "semanticscholar",
                "query": query,
            }
        )
    return records


def search_openalex(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "per-page": min(limit, 50),
        "select": "id,doi,title,display_name,publication_year,cited_by_count,primary_location,authorships,abstract_inverted_index,open_access",
    }
    data = request_json("https://api.openalex.org/works", params=params)
    records = []
    for work in data.get("results") or []:
        loc = work.get("primary_location") or {}
        src = loc.get("source") or {}
        doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
        abstract = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
        oa = work.get("open_access") or {}
        records.append(
            {
                "title": clean_text(work.get("title") or work.get("display_name")),
                "authors": [
                    ((a.get("author") or {}).get("display_name"))
                    for a in (work.get("authorships") or [])[:20]
                    if (a.get("author") or {}).get("display_name")
                ],
                "year": work.get("publication_year"),
                "venue": src.get("display_name"),
                "doi": doi,
                "pdf_url": oa.get("oa_url") if str(oa.get("oa_url") or "").lower().endswith(".pdf") else None,
                "url": loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else work.get("id")),
                "citations": work.get("cited_by_count"),
                "abstract": clean_text(abstract),
                "source": "openalex",
                "query": query,
            }
        )
    return records


def reconstruct_openalex_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    pairs = []
    for word, positions in index.items():
        for pos in positions:
            pairs.append((int(pos), word))
    return " ".join(word for _, word in sorted(pairs))


def search_crossref(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "query.bibliographic": query,
        "rows": min(limit, 50),
        "select": "DOI,title,author,published-print,published-online,published,container-title,is-referenced-by-count,abstract,URL",
    }
    data = request_json("https://api.crossref.org/works", params=params)
    records = []
    for item in (data.get("message") or {}).get("items") or []:
        year = None
        for key in ("published-print", "published-online", "published"):
            parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
            if parts:
                year = parts[0]
                break
        authors = []
        for author in item.get("author") or []:
            name = " ".join(x for x in [author.get("given"), author.get("family")] if x)
            if name:
                authors.append(name)
        records.append(
            {
                "title": clean_text((item.get("title") or [None])[0]),
                "authors": authors,
                "year": year,
                "venue": clean_text((item.get("container-title") or [None])[0]),
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "citations": item.get("is-referenced-by-count"),
                "abstract": clean_text(re.sub(r"<[^>]+>", " ", item.get("abstract") or "")),
                "source": "crossref",
                "query": query,
            }
        )
    return records


def search_dblp(query: str, limit: int) -> list[dict[str, Any]]:
    params = {"q": query, "format": "json", "h": min(limit, 30)}
    data = request_json("https://dblp.org/search/publ/api", params=params)
    hits = (((data.get("result") or {}).get("hits") or {}).get("hit") or [])
    records = []
    for hit in hits:
        info = hit.get("info") or {}
        authors = info.get("authors", {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors.get("text") or authors.get("@pid")]
        elif isinstance(authors, list):
            authors = [a.get("text") if isinstance(a, dict) else str(a) for a in authors]
        records.append(
            {
                "title": clean_text(info.get("title")),
                "authors": [a for a in authors if a],
                "year": int(info["year"]) if str(info.get("year") or "").isdigit() else None,
                "venue": info.get("venue"),
                "doi": info.get("doi"),
                "url": info.get("ee") or info.get("url"),
                "source": "dblp",
                "query": query,
            }
        )
    return records


def search_ddg(query: str, limit: int) -> list[dict[str, Any]]:
    records = []
    ddg_query = f"{query} paper DOI arXiv"
    for item in DDGS().text(ddg_query, max_results=min(limit, 30)):
        records.append(
            {
                "title": clean_text(item.get("title")),
                "url": item.get("href") or item.get("link"),
                "abstract": clean_text(item.get("body")),
                "source": "ddg",
                "query": ddg_query,
            }
        )
    return records


def search_pubmed(query: str, limit: int) -> list[dict[str, Any]]:
    term = f"{query} AND neural networks"
    esearch = request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": min(limit, 20)},
    )
    ids = (esearch.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return []
    esummary = request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    result = esummary.get("result") or {}
    records = []
    for pmid in ids:
        item = result.get(pmid) or {}
        records.append(
            {
                "title": clean_text(item.get("title")),
                "authors": [a.get("name") for a in item.get("authors") or [] if a.get("name")],
                "year": int(str(item.get("pubdate") or "")[:4]) if str(item.get("pubdate") or "")[:4].isdigit() else None,
                "venue": item.get("fulljournalname") or item.get("source"),
                "doi": next((aid.get("value") for aid in item.get("articleids") or [] if aid.get("idtype") == "doi"), None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "pubmed",
                "query": term,
            }
        )
    return records


SOURCES = {
    "arxiv": search_arxiv,
    "semanticscholar": search_semantic_scholar,
    "openalex": search_openalex,
    "crossref": search_crossref,
    "dblp": search_dblp,
    "ddg": search_ddg,
    "pubmed": search_pubmed,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--sources", default=",".join(SOURCES))
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    plan = {"topic": TOPIC, "depth": 5, "facets": FACETS, "sources": selected_sources}
    (args.out / "query_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (raw_dir / "manual_seed.json").write_text(json.dumps(SEED_RECORDS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest: list[dict[str, Any]] = [{"source": "manual_seed", "status": "ok", "records": len(SEED_RECORDS)}]
    for facet in FACETS:
        query = facet["query"]
        for source in selected_sources:
            func = SOURCES[source]
            status: dict[str, Any] = {"source": source, "facet": facet["id"], "query": query}
            started = time.time()
            try:
                records = func(query, args.limit)
                status.update(status="ok" if records else "empty", records=len(records), seconds=round(time.time() - started, 3))
            except Exception as exc:
                records = []
                status.update(status="failed", error=str(exc), records=0, seconds=round(time.time() - started, 3))
            out_file = raw_dir / f"{source}_{facet['id']}.json"
            out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(status)
            print(f"{source:16s} {facet['id']:34s} {status['status']:8s} {status['records']:3d}")
            time.sleep(float(args.sleep))

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
