#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from textwrap import dedent


ROOT = Path("research_radar/local_entropy_dnn_solution_geometry")
CREATED = "2026-06-20"


def md_frontmatter(title: str, tags: list[str], aliases: list[str], source: str, confidence: str) -> str:
    aliases_yaml = "[" + ", ".join(json.dumps(a, ensure_ascii=False) for a in aliases) + "]"
    tags_yaml = "[" + ", ".join(json.dumps(t, ensure_ascii=False) for t in tags) + "]"
    return dedent(
        f"""\
        ---
        title: {json.dumps(title, ensure_ascii=False)}
        tags: {tags_yaml}
        aliases: {aliases_yaml}
        created: {CREATED}
        source: {json.dumps(source, ensure_ascii=False)}
        confidence: {confidence}
        ---

        """
    )


def write_md(path: Path, title: str, tags: list[str], aliases: list[str], source: str, confidence: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_frontmatter(title, tags, aliases, source, confidence) + body.strip() + "\n", encoding="utf-8")


REFS = [
    {
        "key": "Franz1995RecipesMetastable",
        "type": "article",
        "title": "Recipes for metastable states in spin glasses",
        "authors": ["Silvio Franz", "Giorgio Parisi"],
        "year": 1995,
        "venue": "Journal de Physique I",
        "doi": "10.1051/jp1:1995201",
        "arxiv": "cond-mat/9503167",
        "url": "https://arxiv.org/abs/cond-mat/9503167",
        "pdf": "https://arxiv.org/pdf/cond-mat/9503167",
        "tags": ["seminal", "statistical-physics", "franz-parisi"],
        "summary": "고정된 reference configuration과의 overlap을 제한한 자유에너지 potential을 도입해 metastable state를 해석하는 계보의 출발점이다.",
        "method": "Constrained-overlap free energy / Franz-Parisi potential.",
        "dataset": "Spherical p-spin and related spin-glass models.",
        "finding": "reference 주변의 constrained free-energy profile이 metastable state와 연결된다.",
        "limitation": "신경망 실험이 아니라 spin-glass 모델의 이론적 원형이다.",
        "relation": "사용자의 phi(d) 곡선을 'reference 주변 local volume/free energy profile'로 정당화하는 최상위 이론 배경이다.",
        "cite_for": "Franz-Parisi potential, constrained-overlap local free energy.",
    },
    {
        "key": "Gardner1988SpaceInteractions",
        "type": "article",
        "title": "The space of interactions in neural network models",
        "authors": ["Elizabeth Gardner"],
        "year": 1988,
        "venue": "Journal of Physics A: Mathematical and General",
        "doi": "10.1088/0305-4470/21/1/030",
        "url": "https://doi.org/10.1088/0305-4470/21/1/030",
        "tags": ["seminal", "perceptron", "statistical-physics"],
        "summary": "perceptron의 저장 용량과 solution-space를 통계물리 관점에서 분석한 고전 연구다.",
        "method": "Replica/statistical mechanics analysis of perceptron storage.",
        "dataset": "Random pattern-label associations.",
        "finding": "random constraints 아래 weight-space geometry와 capacity를 정량화했다.",
        "limitation": "현대 DNN과 데이터 구조를 직접 다루지는 않는다.",
        "relation": "사용자의 theory arm이 perceptron validation을 두는 이유를 설명한다.",
        "cite_for": "perceptron solution-space/statistical physics lineage.",
    },
    {
        "key": "Seung1992StatisticalMechanics",
        "type": "article",
        "title": "Statistical mechanics of learning from examples",
        "authors": ["H. Sebastian Seung", "Haim Sompolinsky", "Naftali Tishby"],
        "year": 1992,
        "venue": "Physical Review A",
        "doi": "10.1103/PhysRevA.45.6056",
        "url": "https://doi.org/10.1103/PhysRevA.45.6056",
        "tags": ["seminal", "statistical-physics", "learning-theory"],
        "summary": "examples로부터 학습하는 문제를 volume과 generalization의 통계역학으로 해석했다.",
        "method": "Statistical mechanics of version spaces.",
        "dataset": "Teacher-student random examples.",
        "finding": "학습은 일관된 가설 공간의 volume 수축으로 이해될 수 있다.",
        "limitation": "현대 비선형 DNN의 empirical landscape 측정과는 거리가 있다.",
        "relation": "사용자 연구의 'reference 주변 해공간 부피' 관점을 학습 이론 전통에 연결한다.",
        "cite_for": "version-space volume and learning-from-examples framing.",
    },
    {
        "key": "Baldassi2015SubdominantDense",
        "type": "article",
        "title": "Subdominant dense clusters allow for simple learning and high computational performance in neural networks with discrete synapses",
        "authors": ["Carlo Baldassi", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2015,
        "venue": "Physical Review Letters",
        "doi": "10.1103/PhysRevLett.115.128101",
        "arxiv": "1509.05753",
        "url": "https://arxiv.org/abs/1509.05753",
        "pdf": "https://arxiv.org/pdf/1509.05753",
        "tags": ["seminal", "dense-clusters", "local-entropy"],
        "summary": "dominant isolated solutions와 달리 드물지만 조밀한 solution cluster가 학습 가능성과 robust generalization에 연결될 수 있음을 보였다.",
        "method": "Large-deviation local entropy analysis of discrete-synapse perceptrons.",
        "dataset": "Random patterns in binary/discrete synapse models.",
        "finding": "subdominant dense cluster가 알고리즘적으로 접근 가능하고 perturbation에 강하다.",
        "limitation": "discrete/simple model 중심이며 MNIST/DNN empirical estimator는 아니다.",
        "relation": "사용자 novelty의 가장 강한 prior art 중 하나다. 차별점은 실제 reference-pool DNN/MNIST 측정과 QC-aware shell estimator다.",
        "cite_for": "dense cluster hypothesis and local entropy motivation.",
    },
    {
        "key": "Baldassi2016LocalEntropy",
        "type": "article",
        "title": "Local entropy as a measure for sampling solutions in constraint satisfaction problems",
        "authors": ["Carlo Baldassi", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2016,
        "venue": "Journal of Statistical Mechanics: Theory and Experiment",
        "doi": "10.1088/1742-5468/2016/02/023301",
        "arxiv": "1511.05634",
        "url": "https://arxiv.org/abs/1511.05634",
        "pdf": "https://arxiv.org/pdf/1511.05634",
        "tags": ["seminal", "local-entropy", "sampling"],
        "summary": "local entropy를 직접 최적화하거나 추정하여 solution-dense region을 찾는 Entropy-driven Monte Carlo 계열을 제안했다.",
        "method": "Local entropy estimate and entropy-driven Monte Carlo.",
        "dataset": "Binary perceptron and random K-SAT.",
        "finding": "local entropy landscape는 원래 energy landscape보다 알고리즘적으로 더 접근 가능한 신호가 될 수 있다.",
        "limitation": "CSP 중심이며 DNN weight shell에서의 calibrated estimator는 별도 문제다.",
        "relation": "사용자의 shell sampling/phi(d) 측정은 이 계보를 DNN reference 주변의 곡선 추정으로 옮긴다.",
        "cite_for": "local entropy objective and sampling of dense solution regions.",
    },
    {
        "key": "Baldassi2016UnreasonableEffectiveness",
        "type": "article",
        "title": "Unreasonable effectiveness of learning neural networks: From accessible states and robust ensembles to basic algorithmic schemes",
        "authors": ["Carlo Baldassi", "Christian Borgs", "Jennifer Chayes", "Alessandro Ingrosso", "Carlo Lucibello", "Luca Saglietti", "Riccardo Zecchina"],
        "year": 2016,
        "venue": "Proceedings of the National Academy of Sciences",
        "doi": "10.1073/pnas.1608103113",
        "arxiv": "1605.06444",
        "url": "https://arxiv.org/abs/1605.06444",
        "pdf": "https://arxiv.org/pdf/1605.06444",
        "tags": ["seminal", "robust-ensemble", "dense-clusters"],
        "summary": "robust ensemble을 통해 isolated solution을 억제하고 dense accessible state를 강화하는 알고리즘적 틀을 제시했다.",
        "method": "Replica/robust ensemble and algorithmic schemes.",
        "dataset": "Discrete-weight neural models and related optimization problems.",
        "finding": "rare but dense states can be accessible and useful for learning.",
        "limitation": "직접적인 modern DNN empirical local-entropy map은 아니다.",
        "relation": "사용자의 reference-pool approach는 robust/dense state 가설을 실제 trained reference들의 radial profile로 검사한다.",
        "cite_for": "robust ensemble and accessible dense states.",
    },
    {
        "key": "Chaudhari2017EntropySGD",
        "type": "inproceedings",
        "title": "Entropy-SGD: Biasing gradient descent into wide valleys",
        "authors": ["Pratik Chaudhari", "Anna Choromanska", "Stefano Soatto", "Yann LeCun", "Carlo Baldassi", "Christian Borgs", "Jennifer Chayes", "Levent Sagun", "Riccardo Zecchina"],
        "year": 2017,
        "venue": "International Conference on Learning Representations",
        "arxiv": "1611.01838",
        "url": "https://arxiv.org/abs/1611.01838",
        "pdf": "https://arxiv.org/pdf/1611.01838",
        "tags": ["sota", "optimization", "local-entropy"],
        "summary": "inner-loop Langevin dynamics로 local entropy gradient를 추정해 wide valley를 선호하도록 SGD를 편향한다.",
        "method": "Local-entropy objective optimized by nested SGD/Langevin dynamics.",
        "dataset": "CNN/RNN benchmarks in the original experiments.",
        "finding": "local entropy objective가 smoother landscape와 generalization improvement를 보일 수 있다.",
        "limitation": "학습 알고리즘 제안이지, fixed reference 주변 phi(d) 곡선의 QC-aware 측정은 아니다.",
        "relation": "사용자의 방법은 optimization이 아니라 측정/진단에 초점을 둔다는 점에서 차별화된다.",
        "cite_for": "local-entropy objective in deep-learning optimization.",
    },
    {
        "key": "Baldassi2020ShapingLandscape",
        "type": "article",
        "title": "Shaping the learning landscape in neural networks around wide flat minima",
        "authors": ["Carlo Baldassi", "Fabrizio Pittorino", "Riccardo Zecchina"],
        "year": 2020,
        "venue": "Proceedings of the National Academy of Sciences",
        "doi": "10.1073/pnas.1908636117",
        "arxiv": "1905.07833",
        "url": "https://arxiv.org/abs/1905.07833",
        "pdf": "https://arxiv.org/pdf/1905.07833",
        "tags": ["sota", "wide-flat-minima", "statistical-physics"],
        "summary": "one/two-layer neural models에서 wide flat minima의 존재와 cross-entropy 학습의 연결을 분석했다.",
        "method": "Statistical mechanics and numerical study of simple neural-network landscapes.",
        "dataset": "Random patterns and real-data numerical checks.",
        "finding": "wide flat minima coexist with narrower minima and correlate with robustness/generalization.",
        "limitation": "사용자처럼 MNIST rule complexity와 reference-pool radial estimator를 결합하지는 않는다.",
        "relation": "가장 가까운 prior art. 사용자 연구의 novelty는 theory validation + empirical rule complexity axis + QC-aware sampling package다.",
        "cite_for": "wide flat minima in neural-network landscape.",
    },
    {
        "key": "Baldassi2021UnveilingStructure",
        "type": "article",
        "title": "Unveiling the structure of wide flat minima in neural networks",
        "authors": ["Carlo Baldassi", "Clarissa Lauditi", "Enrico M. Malatesta", "Gabriele Perugini", "Riccardo Zecchina"],
        "year": 2021,
        "venue": "Physical Review Letters",
        "doi": "10.1103/PhysRevLett.127.278301",
        "arxiv": "2107.01163",
        "url": "https://arxiv.org/abs/2107.01163",
        "pdf": "https://arxiv.org/pdf/2107.01163",
        "tags": ["sota", "wide-flat-minima", "high-margin"],
        "summary": "wide flat minima가 high-margin solutions 주변의 extensive structures로 나타난다고 분석했다.",
        "method": "Analytical/numerical statistical-physics analysis.",
        "dataset": "Neural-network models with random patterns.",
        "finding": "rare high-margin minima are surrounded by many lower-margin solutions over long distances.",
        "limitation": "task-specific empirical DNN reference families and estimator QC are outside scope.",
        "relation": "사용자 phi(d) profile에서 거리별 local support shape를 해석할 때 핵심 prior다.",
        "cite_for": "structure of wide flat minima and high-margin cores.",
    },
    {
        "key": "Baldassi2021EntropicGradient",
        "type": "article",
        "title": "Entropic gradient descent algorithms and wide flat minima",
        "authors": ["Carlo Baldassi", "Fabrizio Pittorino", "Riccardo Zecchina"],
        "year": 2021,
        "venue": "Journal of Statistical Mechanics: Theory and Experiment",
        "doi": "10.1088/1742-5468/ac3ae8",
        "url": "https://doi.org/10.1088/1742-5468/ac3ae8",
        "tags": ["method", "local-entropy", "optimization"],
        "summary": "entropic-gradient 방식으로 wide flat minima를 겨냥하는 알고리즘 계열을 정리한다.",
        "method": "Entropy-biased gradient methods.",
        "dataset": "Simple neural models.",
        "finding": "entropy signal can guide optimization toward wide flat minima.",
        "limitation": "사용자처럼 고정 reference의 shell partition function을 실험적으로 스캔하지 않는다.",
        "relation": "optimization prior art와 measurement-focused contribution을 구분할 때 유용하다.",
        "cite_for": "entropic gradient and algorithmic route to wide minima.",
    },
    {
        "key": "Pittorino2022DeepNetworksToroids",
        "type": "inproceedings",
        "title": "Deep networks on toroids: removing symmetries reveals the structure of flat regions in the landscape geometry",
        "authors": ["Fabrizio Pittorino", "Antonio Ferraro", "Gabriele Perugini", "Carlo Baldassi", "Riccardo Zecchina"],
        "year": 2022,
        "venue": "International Conference on Machine Learning",
        "doi": "10.1088/1742-5468/ac9832",
        "arxiv": "2202.03038",
        "url": "https://proceedings.mlr.press/v162/pittorino22a.html",
        "pdf": "https://proceedings.mlr.press/v162/pittorino22a/pittorino22a.pdf",
        "tags": ["sota", "symmetry", "flat-regions"],
        "summary": "parameter symmetry를 제거해야 flat-region geometry가 더 의미 있게 보인다는 문제를 제기한다.",
        "method": "Symmetry-aware landscape analysis.",
        "dataset": "Neural-network models and empirical checks.",
        "finding": "symmetry quotienting changes the apparent geometry of flat regions.",
        "limitation": "local radial shell estimator나 dataset-complexity ordering 자체가 주제는 아니다.",
        "relation": "사용자 방법의 한계: raw parameter distance의 symmetry sensitivity를 명시해야 한다.",
        "cite_for": "parameter symmetries as a caveat for flatness/local geometry.",
    },
    {
        "key": "Baldassi2022LearningAtypical",
        "type": "article",
        "title": "Learning through atypical phase transitions in overparameterized neural networks",
        "authors": ["Carlo Baldassi", "Clarissa Lauditi", "Enrico M. Malatesta", "Rosalba Pacelli", "Gabriele Perugini", "Riccardo Zecchina"],
        "year": 2022,
        "venue": "Physical Review E",
        "doi": "10.1103/PhysRevE.106.014116",
        "arxiv": "2110.00683",
        "url": "https://arxiv.org/abs/2110.00683",
        "pdf": "https://arxiv.org/pdf/2110.00683",
        "tags": ["sota", "phase-transition", "overparameterization"],
        "summary": "overparameterized neural models에서 typical interpolation과 atypical wide-region transition을 구분한다.",
        "method": "Statistical mechanics of overparameterized binary neural-network models.",
        "dataset": "Teacher-student/random-rule settings.",
        "finding": "efficient algorithms may sample atypical rare regions rather than exponentially dominant typical solutions.",
        "limitation": "MNIST rule-family empirical mapping is not the focus.",
        "relation": "사용자의 random/teacher/low-TV rule 비교 해석에 직접적인 이론적 선행근거다.",
        "cite_for": "atypical phase transition and algorithmically relevant wide regions.",
    },
    {
        "key": "Baldassi2023TypicalAtypical",
        "type": "article",
        "title": "Typical and atypical solutions in nonconvex neural networks with discrete and continuous weights",
        "authors": ["Carlo Baldassi", "Enrico M. Malatesta", "Gabriele Perugini", "Riccardo Zecchina"],
        "year": 2023,
        "venue": "Physical Review E",
        "doi": "10.1103/PhysRevE.108.024310",
        "arxiv": "2304.13871",
        "url": "https://arxiv.org/abs/2304.13871",
        "pdf": "https://arxiv.org/pdf/2304.13871",
        "tags": ["sota", "typical-atypical", "wide-flat-minima"],
        "summary": "discrete/continuous nonconvex neural models에서 typical and atypical solution geometry를 비교한다.",
        "method": "1RSB/landscape analysis and numerical evidence.",
        "dataset": "Negative-margin perceptron variants.",
        "finding": "wide flat minimizers coexist with dominant narrow/hierarchical background solutions.",
        "limitation": "실제 trained MNIST reference pool에 대한 estimator가 아니다.",
        "relation": "사용자 연구의 reference family stratification과 closest theoretical framing을 제공한다.",
        "cite_for": "typical/atypical solution geometry.",
    },
    {
        "key": "Barbier2024AtypicalSolutions",
        "type": "article",
        "title": "On the atypical solutions of the symmetric binary perceptron",
        "authors": ["Damien Barbier", "Ahmed El Alaoui", "Florent Krzakala", "Lenka Zdeborová"],
        "year": 2024,
        "venue": "Journal of Physics A: Mathematical and Theoretical",
        "arxiv": "2310.02850",
        "url": "https://arxiv.org/abs/2310.02850",
        "pdf": "https://arxiv.org/pdf/2310.02850",
        "tags": ["sota", "perceptron", "franz-parisi"],
        "summary": "symmetric binary perceptron에서 rare high-margin solutions의 local entropy/Franz-Parisi potential을 분석한다.",
        "method": "First/second moment methods plus replica analysis under assumptions.",
        "dataset": "Symmetric binary perceptron.",
        "finding": "rare solutions may have extensive-entropy clusters and entropic/energetic barriers.",
        "limitation": "작은 constraint-density regime와 binary perceptron에 특화되어 있다.",
        "relation": "사용자 theory validation의 더 최신 perceptron-side context다.",
        "cite_for": "recent Franz-Parisi/local-entropy analysis of atypical perceptron solutions.",
    },
    {
        "key": "Dinh2017SharpMinima",
        "type": "inproceedings",
        "title": "Sharp minima can generalize for deep nets",
        "authors": ["Laurent Dinh", "Razvan Pascanu", "Samy Bengio", "Yoshua Bengio"],
        "year": 2017,
        "venue": "International Conference on Machine Learning",
        "arxiv": "1703.04933",
        "url": "https://arxiv.org/abs/1703.04933",
        "pdf": "https://arxiv.org/pdf/1703.04933",
        "tags": ["limitation", "flatness", "reparameterization"],
        "summary": "deep nets에서는 flatness/sharpness measure가 reparameterization에 의해 조작될 수 있음을 보였다.",
        "method": "Analytical counterexamples based on parameter-space symmetries.",
        "dataset": "Deep ReLU network settings.",
        "finding": "naive parameter-space sharpness is not a reliable invariant explanation of generalization.",
        "limitation": "local entropy volume 자체를 완전히 부정하지는 않으며, 측정 좌표계와 regularization을 요구한다.",
        "relation": "사용자의 raw L2-shell 해석에서 반드시 언급해야 하는 주요 위협이다.",
        "cite_for": "caveat about reparameterization-sensitive flatness.",
    },
    {
        "key": "Keskar2017LargeBatch",
        "type": "inproceedings",
        "title": "On large-batch training for deep learning: Generalization gap and sharp minima",
        "authors": ["Nitish Shirish Keskar", "Dheevatsa Mudigere", "Jorge Nocedal", "Mikhail Smelyanskiy", "Ping Tak Peter Tang"],
        "year": 2017,
        "venue": "International Conference on Learning Representations",
        "arxiv": "1609.04836",
        "url": "https://arxiv.org/abs/1609.04836",
        "pdf": "https://arxiv.org/pdf/1609.04836",
        "tags": ["seminal", "sharpness", "generalization"],
        "summary": "large-batch training과 sharp minima/generalization gap의 경험적 연결을 대중화했다.",
        "method": "Empirical comparison of batch size and sharpness.",
        "dataset": "Standard deep-learning benchmarks.",
        "finding": "large batches can converge to sharper minima with worse generalization in studied settings.",
        "limitation": "flatness metric invariance와 causality는 후속 연구에서 강하게 논쟁되었다.",
        "relation": "local entropy/flatness debate의 empirical entry point로 쓰인다.",
        "cite_for": "sharp-minima generalization-gap hypothesis.",
    },
    {
        "key": "Jiang2020FantasticGeneralization",
        "type": "inproceedings",
        "title": "Fantastic generalization measures and where to find them",
        "authors": ["Yiding Jiang", "Behnam Neyshabur", "Hossein Mobahi", "Dilip Krishnan", "Samy Bengio"],
        "year": 2020,
        "venue": "International Conference on Learning Representations",
        "arxiv": "1912.02178",
        "url": "https://arxiv.org/abs/1912.02178",
        "pdf": "https://arxiv.org/pdf/1912.02178",
        "tags": ["survey", "generalization-measures", "limitation"],
        "summary": "40개 이상의 generalization measure를 대규모 실험에서 비교해 일부 measure의 실패와 유망한 방향을 보였다.",
        "method": "Large-scale controlled empirical study.",
        "dataset": "Over 10,000 trained convolutional networks.",
        "finding": "generalization measure는 작은 실험만으로 주장하기 어렵고, controlled variation이 필요하다.",
        "limitation": "local entropy shell estimator 자체를 직접 검증하지는 않는다.",
        "relation": "사용자의 novelty claim을 'generalization 설명'보다 'diagnostic measure'로 제한해야 한다는 근거다.",
        "cite_for": "caution around generalization measures.",
    },
    {
        "key": "Andriushchenko2023ModernSharpness",
        "type": "inproceedings",
        "title": "A modern look at the relationship between sharpness and generalization",
        "authors": ["Maksym Andriushchenko", "Francesco Croce", "Maximilian Müller", "Matthias Hein", "Nicolas Flammarion"],
        "year": 2023,
        "venue": "International Conference on Machine Learning",
        "doi": "10.5555/3618408.3618444",
        "arxiv": "2302.07011",
        "url": "https://arxiv.org/abs/2302.07011",
        "pdf": "https://arxiv.org/pdf/2302.07011",
        "tags": ["negative", "sharpness", "modern-settings"],
        "summary": "modern architectures/settings에서 sharpness-generalization correlation이 일관적이지 않음을 실험적으로 보였다.",
        "method": "Empirical study across ConvNets, transformers, fine-tuning settings.",
        "dataset": "ImageNet/CIFAR/CLIP/BERT style settings.",
        "finding": "sharpness can correlate with training hyperparameters and may fail as a universal explanation.",
        "limitation": "local volume/free energy curve와 dataset-rule complexity의 직접 분석은 아니다.",
        "relation": "사용자 논문이 과도한 generalization claim을 피해야 하는 핵심 근거다.",
        "cite_for": "limitations of sharpness as universal generalization explanation.",
    },
    {
        "key": "Foret2021SharpnessAware",
        "type": "inproceedings",
        "title": "Sharpness-aware minimization for efficiently improving generalization",
        "authors": ["Pierre Foret", "Ariel Kleiner", "Hossein Mobahi", "Behnam Neyshabur"],
        "year": 2021,
        "venue": "International Conference on Learning Representations",
        "arxiv": "2010.01412",
        "url": "https://arxiv.org/abs/2010.01412",
        "pdf": "https://arxiv.org/pdf/2010.01412",
        "tags": ["sota", "sharpness", "optimization"],
        "summary": "parameter neighborhood의 worst-case loss를 줄이는 SAM objective를 제안했다.",
        "method": "Min-max sharpness-aware training objective.",
        "dataset": "Vision/NLP benchmark experiments.",
        "finding": "neighborhood-aware optimization can improve generalization in many settings.",
        "limitation": "SAM은 training objective이며 local entropy density profile estimation과 다르다.",
        "relation": "사용자 방법을 optimization method가 아닌 measurement protocol로 구분할 때 필요하다.",
        "cite_for": "neighborhood-aware optimization baseline.",
    },
    {
        "key": "Dziugaite2017ComputingNonvacuous",
        "type": "inproceedings",
        "title": "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data",
        "authors": ["Gintare Karolina Dziugaite", "Daniel M. Roy"],
        "year": 2017,
        "venue": "Conference on Uncertainty in Artificial Intelligence",
        "arxiv": "1703.11008",
        "url": "https://arxiv.org/abs/1703.11008",
        "pdf": "https://arxiv.org/pdf/1703.11008",
        "tags": ["pac-bayes", "generalization", "flatness"],
        "summary": "PAC-Bayes posterior를 학습해 overparameterized net에 nonvacuous bound를 계산했다.",
        "method": "PAC-Bayes bound optimization.",
        "dataset": "MNIST-style experiments.",
        "finding": "stochastic neural-network posterior와 generalization bounds를 실제로 계산할 수 있음을 보였다.",
        "limitation": "local radial density curve를 직접 측정하지 않는다.",
        "relation": "사용자 MNIST local entropy를 generalization bound와 연결할 때 참고하되, 동일 주장으로 과장하면 안 된다.",
        "cite_for": "PAC-Bayes posterior/flatness relation.",
    },
    {
        "key": "Haddouche2025PACBayesianLink",
        "type": "inproceedings",
        "title": "A PAC-Bayesian link between generalisation and flat minima",
        "authors": ["Maxime Haddouche", "Paul Viallard", "Umut Simsekli", "Benjamin Guedj"],
        "year": 2025,
        "venue": "Proceedings of Machine Learning Research, ALT",
        "arxiv": "2402.08508",
        "url": "https://proceedings.mlr.press/v272/haddouche25a.html",
        "pdf": "https://raw.githubusercontent.com/mlresearch/v272/main/assets/haddouche25a/haddouche25a.pdf",
        "tags": ["recent", "pac-bayes", "flat-minima"],
        "summary": "Poincare/Log-Sobolev inequalities와 PAC-Bayes를 결합해 flat minima와 generalization의 formal link를 제시한다.",
        "method": "PAC-Bayes bounds with gradient/functional inequalities.",
        "dataset": "Theoretical analysis.",
        "finding": "dimension-explicit dependence를 피하는 방식으로 flat minima의 긍정적 효과를 formalize한다.",
        "limitation": "사용자 실험의 phi(d) estimator 검정과 직접 동일하지는 않다.",
        "relation": "Discussion에서 local entropy 결과의 이론적 함의를 조심스럽게 연결할 수 있다.",
        "cite_for": "recent PAC-Bayes formal link to flat minima.",
    },
    {
        "key": "Zhang2017RethinkingGeneralization",
        "type": "inproceedings",
        "title": "Understanding deep learning requires rethinking generalization",
        "authors": ["Chiyuan Zhang", "Samy Bengio", "Moritz Hardt", "Benjamin Recht", "Oriol Vinyals"],
        "year": 2017,
        "venue": "International Conference on Learning Representations",
        "arxiv": "1611.03530",
        "url": "https://arxiv.org/abs/1611.03530",
        "pdf": "https://arxiv.org/pdf/1611.03530",
        "tags": ["seminal", "random-labels", "generalization"],
        "summary": "deep nets가 random labels도 fit할 수 있음을 보여 일반화 설명에 데이터/알고리즘/implicit bias가 필요함을 제기했다.",
        "method": "Controlled label randomization experiments.",
        "dataset": "Image benchmarks with true/random labels.",
        "finding": "capacity alone cannot explain generalization.",
        "limitation": "random labels의 local entropy profile을 직접 측정하지 않는다.",
        "relation": "사용자 MNIST random_label rule이 왜 중요한 negative/control axis인지 설명한다.",
        "cite_for": "random labels and memorization challenge.",
    },
    {
        "key": "Arpit2017CloserMemorization",
        "type": "inproceedings",
        "title": "A closer look at memorization in deep networks",
        "authors": ["Devansh Arpit", "Stanislaw Jastrzebski", "Nicolas Ballas", "David Krueger", "Emmanuel Bengio", "Maxinder S. Kanwal", "Tegan Maharaj", "Asja Fischer", "Aaron Courville", "Yoshua Bengio", "Simon Lacoste-Julien"],
        "year": 2017,
        "venue": "International Conference on Machine Learning",
        "arxiv": "1706.05394",
        "url": "https://arxiv.org/abs/1706.05394",
        "pdf": "https://arxiv.org/pdf/1706.05394",
        "tags": ["dataset", "memorization", "random-labels"],
        "summary": "DNN은 simple pattern을 먼저 학습하고 noisy/random labels를 더 늦게 memorization하는 경향을 보인다고 분석했다.",
        "method": "Empirical training dynamics and noisy-label experiments.",
        "dataset": "Vision benchmarks and random/noisy labels.",
        "finding": "memorization and generalization can be separated in training dynamics.",
        "limitation": "solution-space local entropy 측정이 아니라 training behavior 분석이다.",
        "relation": "사용자 rule complexity ordering을 memorization/generalization 맥락에 연결한다.",
        "cite_for": "memorization vs structured pattern learning.",
    },
    {
        "key": "Li2018VisualizingLoss",
        "type": "inproceedings",
        "title": "Visualizing the loss landscape of neural nets",
        "authors": ["Hao Li", "Zheng Xu", "Gavin Taylor", "Christoph Studer", "Tom Goldstein"],
        "year": 2018,
        "venue": "Advances in Neural Information Processing Systems",
        "arxiv": "1712.09913",
        "url": "https://arxiv.org/abs/1712.09913",
        "pdf": "https://arxiv.org/pdf/1712.09913",
        "tags": ["method", "loss-landscape", "visualization"],
        "summary": "filter normalization 등을 포함한 loss landscape visualization 방법을 제시했다.",
        "method": "Low-dimensional visualization of loss surfaces.",
        "dataset": "Standard deep-learning benchmarks.",
        "finding": "architecture/optimization choices produce visibly different loss-surface geometry.",
        "limitation": "2D visualization은 high-dimensional shell volume을 직접 측정하지 않는다.",
        "relation": "사용자 방법을 visualization이 아니라 radial partition/profile estimator로 차별화한다.",
        "cite_for": "loss landscape visualization baseline.",
    },
    {
        "key": "Garipov2018LossSurfaces",
        "type": "inproceedings",
        "title": "Loss surfaces, mode connectivity, and fast ensembling of DNNs",
        "authors": ["Timur Garipov", "Pavel Izmailov", "Dmitrii Podoprikhin", "Dmitry Vetrov", "Andrew Gordon Wilson"],
        "year": 2018,
        "venue": "Advances in Neural Information Processing Systems",
        "arxiv": "1802.10026",
        "url": "https://arxiv.org/abs/1802.10026",
        "pdf": "https://arxiv.org/pdf/1802.10026",
        "tags": ["landscape", "mode-connectivity", "adjacent"],
        "summary": "independently trained solutions가 low-loss paths로 연결될 수 있음을 보이고 fast ensembling에 활용했다.",
        "method": "Curve finding in parameter space.",
        "dataset": "Standard DNN benchmarks.",
        "finding": "apparent minima are often connected by low-loss manifolds.",
        "limitation": "local radial density around one reference를 정량화하지 않는다.",
        "relation": "사용자의 radial profile이 global connectivity와 다른 정보임을 설명할 때 사용한다.",
        "cite_for": "mode connectivity and non-isolated minima.",
    },
    {
        "key": "Draxler2018EssentiallyNoBarriers",
        "type": "inproceedings",
        "title": "Essentially no barriers in neural network energy landscape",
        "authors": ["Felix Draxler", "Kambis Veschgini", "Manfred Salmhofer", "Fred A. Hamprecht"],
        "year": 2018,
        "venue": "International Conference on Machine Learning",
        "arxiv": "1803.00885",
        "url": "https://arxiv.org/abs/1803.00885",
        "pdf": "https://arxiv.org/pdf/1803.00885",
        "tags": ["landscape", "mode-connectivity", "adjacent"],
        "summary": "neural network solutions 사이의 low-loss path를 empirical하게 보였다.",
        "method": "Path finding and loss landscape exploration.",
        "dataset": "DNN classification benchmarks.",
        "finding": "energy barriers between minima can be small in overparameterized nets.",
        "limitation": "reference-local shell entropy와 global path connectivity는 다른 측정이다.",
        "relation": "local entropy 결과를 isolated basin 주장으로 과장하지 않게 해준다.",
        "cite_for": "low-barrier connectivity caveat.",
    },
    {
        "key": "Izmailov2018AveragingWeights",
        "type": "inproceedings",
        "title": "Averaging weights leads to wider optima and better generalization",
        "authors": ["Pavel Izmailov", "Dmitrii Podoprikhin", "Timur Garipov", "Dmitry Vetrov", "Andrew Gordon Wilson"],
        "year": 2018,
        "venue": "Conference on Uncertainty in Artificial Intelligence",
        "arxiv": "1803.05407",
        "url": "https://arxiv.org/abs/1803.05407",
        "pdf": "https://arxiv.org/pdf/1803.05407",
        "tags": ["optimization", "flatness", "swa"],
        "summary": "stochastic weight averaging이 wider optima와 더 나은 generalization을 유도할 수 있음을 보였다.",
        "method": "Weight averaging along SGD trajectory.",
        "dataset": "Vision benchmarks.",
        "finding": "SGD trajectory samples can be averaged to land in flatter regions.",
        "limitation": "volume profile의 absolute partition function 추정은 아니다.",
        "relation": "local entropy measurement와 optimization trajectory-based flatness를 구분한다.",
        "cite_for": "wider optima through stochastic weight averaging.",
    },
    {
        "key": "Mele2025DensityStates",
        "type": "article",
        "title": "Density of states in neural networks: an in-depth exploration of learning in parameter space",
        "authors": ["Margherita Mele", "Roberto Menichetti", "Alessandro Ingrosso", "Raffaello Potestio"],
        "year": 2025,
        "venue": "Transactions on Machine Learning Research",
        "arxiv": "2409.18683",
        "url": "https://openreview.net/forum?id=BLDtWlFKhn",
        "pdf": "https://openreview.net/pdf?id=BLDtWlFKhn",
        "tags": ["recent", "density-of-states", "dataset-complexity"],
        "summary": "Wang-Landau sampling으로 neural network density of states를 추정하고 데이터 구조와 loss spectrum의 관계를 분석했다.",
        "method": "Wang-Landau enhanced sampling of density of states.",
        "dataset": "Real-world and synthetic datasets with binary-state networks.",
        "finding": "dataset structure affects the density of states across loss values.",
        "limitation": "reference-centered radial local entropy가 아니라 global loss-spectrum density다.",
        "relation": "사용자 novelty의 강한 인접 prior art다. 차별점은 reference-local phi(d), MNIST rule families, shell QC다.",
        "cite_for": "recent density-of-states approach linking data structure and parameter-space geometry.",
    },
    {
        "key": "Winer2026DeepNeuralNetsHamiltonians",
        "type": "article",
        "title": "Deep neural nets as Hamiltonians",
        "authors": ["Mike Winer", "Boris Hanin"],
        "year": 2026,
        "venue": "Physical Review E",
        "doi": "10.1103/xg33-ksqn",
        "arxiv": "2503.23982",
        "url": "https://doi.org/10.1103/xg33-ksqn",
        "tags": ["recent", "statistical-physics", "hamiltonian"],
        "summary": "DNN을 Hamiltonian 관점에서 해석해 output distribution과 statistical physics tools의 접점을 넓힌다.",
        "method": "Theoretical analysis of network-output distributions as Hamiltonians.",
        "dataset": "Theoretical/statistical mechanics setting.",
        "finding": "replica/Franz-Parisi 계열 도구가 modern neural-net theory에 계속 확장되고 있음을 보여준다.",
        "limitation": "사용자의 shell estimator와 직접 같은 quantity는 아니다.",
        "relation": "최근 연구지형에서 statistical-physics framing이 계속 살아 있음을 보여주는 context다.",
        "cite_for": "recent statistical-physics framing of deep nets.",
    },
    {
        "key": "Neal2001AnnealedImportance",
        "type": "article",
        "title": "Annealed importance sampling",
        "authors": ["Radford M. Neal"],
        "year": 2001,
        "venue": "Statistics and Computing",
        "doi": "10.1023/A:1008923215028",
        "arxiv": "physics/9803008",
        "url": "https://arxiv.org/abs/physics/9803008",
        "pdf": "https://arxiv.org/pdf/physics/9803008",
        "tags": ["method", "sampling", "partition-function"],
        "summary": "annealing path를 통해 normalizing constant를 추정하는 importance sampling 방법이다.",
        "method": "Annealed importance sampling.",
        "dataset": "Generic probabilistic models.",
        "finding": "isolated modes and normalizing constants in high dimension can be handled with annealing sequences.",
        "limitation": "shell geometry/vMF proposal을 직접 다루지 않는다.",
        "relation": "사용자 logZ/partition estimation method의 broader Monte Carlo foundation이다.",
        "cite_for": "normalizing-constant estimation by annealed importance sampling.",
    },
    {
        "key": "DelMoral2006SMCSamplers",
        "type": "article",
        "title": "Sequential Monte Carlo samplers",
        "authors": ["Pierre Del Moral", "Arnaud Doucet", "Ajay Jasra"],
        "year": 2006,
        "venue": "Journal of the Royal Statistical Society: Series B",
        "doi": "10.1111/j.1467-9868.2006.00553.x",
        "url": "https://doi.org/10.1111/j.1467-9868.2006.00553.x",
        "pdf": "https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf",
        "tags": ["method", "smc", "partition-function"],
        "summary": "공통 공간 위 분포열을 sequentially sample하는 SMC samplers의 표준 이론을 제시했다.",
        "method": "Sequential Monte Carlo over distribution sequences.",
        "dataset": "Generic Bayesian/statistical models.",
        "finding": "known-up-to-constant target distributions can be approximated by weighted particle systems.",
        "limitation": "DNN local entropy estimator의 구체 구현은 아니다.",
        "relation": "사용자 PM-SAIS fallback/QC policy의 방법론적 근거다.",
        "cite_for": "SMC sampler foundation for log normalizer estimation.",
    },
    {
        "key": "Wood1994SimulationVMF",
        "type": "article",
        "title": "Simulation of the von Mises Fisher distribution",
        "authors": ["Andrew T. A. Wood"],
        "year": 1994,
        "venue": "Communications in Statistics - Simulation and Computation",
        "doi": "10.1080/03610919408813161",
        "url": "https://doi.org/10.1080/03610919408813161",
        "tags": ["method", "vmf", "directional-statistics"],
        "summary": "hypersphere 위 von Mises-Fisher 분포 샘플링 알고리즘을 제시한다.",
        "method": "vMF random variate simulation.",
        "dataset": "Directional distributions on spheres.",
        "finding": "high-dimensional unit-vector proposals can be sampled from mean-direction concentrated distributions.",
        "limitation": "importance weighting/logZ estimation은 별도 설계가 필요하다.",
        "relation": "사용자 shell sampler의 vMF-centered proposal 구현 근거다.",
        "cite_for": "vMF shell proposal sampling.",
    },
    {
        "key": "Shuman2013EmergingGraphSignal",
        "type": "article",
        "title": "The emerging field of signal processing on graphs",
        "authors": ["David I. Shuman", "Sunil K. Narang", "Pascal Frossard", "Antonio Ortega", "Pierre Vandergheynst"],
        "year": 2013,
        "venue": "IEEE Signal Processing Magazine",
        "doi": "10.1109/MSP.2012.2235192",
        "url": "https://doi.org/10.1109/MSP.2012.2235192",
        "tags": ["dataset", "graph-tv", "complexity"],
        "summary": "graph signal smoothness, variation, spectral tools의 표준적인 개관이다.",
        "method": "Graph signal processing framework.",
        "dataset": "Generic graph signals.",
        "finding": "graph total variation/smoothness can quantify signal variation over data geometry.",
        "limitation": "neural network local entropy와 직접 연결하지 않는다.",
        "relation": "사용자의 NMSTV label-complexity axis를 정당화하는 인접 분야 근거다.",
        "cite_for": "graph total variation and label smoothness framing.",
    },
    {
        "key": "Ortega2018GraphSignalProcessing",
        "type": "article",
        "title": "Graph signal processing: Overview, challenges, and applications",
        "authors": ["Antonio Ortega", "Pascal Frossard", "Jelena Kovačević", "José M. F. Moura", "Pierre Vandergheynst"],
        "year": 2018,
        "venue": "Proceedings of the IEEE",
        "doi": "10.1109/JPROC.2018.2820126",
        "url": "https://doi.org/10.1109/JPROC.2018.2820126",
        "tags": ["dataset", "graph-tv", "complexity"],
        "summary": "graph signal processing의 응용과 challenge를 폭넓게 정리한다.",
        "method": "Survey/tutorial.",
        "dataset": "Graph-structured data applications.",
        "finding": "data geometry 위 smoothness/variation은 signal complexity를 표현하는 핵심 개념이다.",
        "limitation": "DNN weight-space geometry는 다루지 않는다.",
        "relation": "사용자 MNIST rule complexity를 label graph signal로 설명할 수 있게 해준다.",
        "cite_for": "graph signal processing and smoothness measures.",
    },
    {
        "key": "Zhou2004LearningLocalGlobal",
        "type": "inproceedings",
        "title": "Learning with local and global consistency",
        "authors": ["Dengyong Zhou", "Olivier Bousquet", "Thomas N. Lal", "Jason Weston", "Bernhard Schölkopf"],
        "year": 2004,
        "venue": "Advances in Neural Information Processing Systems",
        "url": "https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency",
        "tags": ["dataset", "graph-smoothness", "semi-supervised-learning"],
        "summary": "graph 위 local/global consistency로 labels를 propagate하는 고전 연구다.",
        "method": "Graph-based semi-supervised learning.",
        "dataset": "Benchmark graph/data classification tasks.",
        "finding": "nearby examples should have consistent labels라는 graph smoothness assumption을 formalize한다.",
        "limitation": "weight-space local entropy와 직접 관계는 없다.",
        "relation": "사용자 low-TV/even-odd/random label rules의 label smoothness interpretation을 보조한다.",
        "cite_for": "graph label smoothness/local consistency assumption.",
    },
]


def bibtex_entry(ref: dict) -> str:
    typ = ref["type"]
    fields = {
        "title": ref["title"],
        "author": " and ".join(ref["authors"]),
        "year": str(ref["year"]),
    }
    if typ == "article":
        fields["journal"] = ref["venue"]
    else:
        fields["booktitle"] = ref["venue"]
    if ref.get("doi"):
        fields["doi"] = ref["doi"]
    if ref.get("arxiv"):
        fields["eprint"] = ref["arxiv"]
        fields["archivePrefix"] = "arXiv"
    if ref.get("url"):
        fields["url"] = ref["url"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@{typ}{{{ref['key']},\n{body}\n}}\n"


def generate_bib() -> None:
    (ROOT / "07_Bibliography.bib").write_text("\n".join(bibtex_entry(r) for r in REFS), encoding="utf-8")
    corpus = []
    for i, r in enumerate(REFS, 1):
        corpus.append(
            {
                "rank": i,
                "bibkey": r["key"],
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "venue": r["venue"],
                "doi": r.get("doi"),
                "arxiv_id": r.get("arxiv"),
                "url": r.get("url"),
                "pdf_url": r.get("pdf"),
                "source": "curated_verified",
                "query": "local entropy neural network solution geometry",
            }
        )
    (ROOT / "megasearch" / "curated_verified_corpus_for_pdfs.json").write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    top20_keys = pdf_top20_keys()
    top20 = []
    by_key = {r["key"]: r for r in REFS}
    for i, key in enumerate(top20_keys, 1):
        r = by_key[key]
        top20.append(
            {
                "rank": i,
                "bibkey": r["key"],
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "venue": r["venue"],
                "doi": r.get("doi"),
                "arxiv_id": r.get("arxiv"),
                "url": r.get("url"),
                "pdf_url": r.get("pdf"),
                "source": "curated_verified_top20",
                "query": "must read local entropy DNN solution geometry",
            }
        )
    (ROOT / "megasearch" / "pdf_top20_corpus.json").write_text(
        json.dumps(top20, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def paper_cards() -> None:
    for r in REFS:
        links = [f"[[maps/{tag.replace('-', '_')}]]" for tag in r["tags"][:2]]
        body = f"""
# {r['title']}

## Metadata

- bibkey: [@{r['key']}]
- authors: {', '.join(r['authors'])}
- year: {r['year']}
- venue: {r['venue']}
- DOI/arXiv/URL: {r.get('doi', 'n/a')} / {r.get('arxiv', 'n/a')} / {r.get('url', 'n/a')}
- evidence_basis: metadata + abstract/PDF where available
- tags: {', '.join(r['tags'])}

## Summary

{r['summary']}

## Method

{r['method']}

## Dataset

{r['dataset']}

## Key Finding

{r['finding']}

## Limitation

{r['limitation']}

## Relation To My Work

{r['relation']}

## Cite For

{r['cite_for']}

## Backlinks

- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- {' '.join(links)}
"""
        write_md(
            ROOT / "papers" / f"{r['key']}.md",
            r["title"],
            ["paper-card"] + r["tags"],
            [r["key"], r["title"]],
            "curated bibliography; DOI/arXiv/OpenReview/PMLR/venue metadata",
            "high",
            body,
        )


def lab_cards() -> None:
    lab_root = ROOT / "lab_radar"
    for report in sorted(lab_root.glob("*/report.md")):
        slug = report.parent.name
        metrics_path = report.parent / "metrics.json"
        meta_path = report.parent / "meta.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        top_topics = metrics.get("top_topics", [])[:8]
        emerging = metrics.get("emerging_topics", [])[:5]
        declining = metrics.get("declining_topics", [])[:5]
        body = f"""
# {slug.replace('_', ' ').title()}

## Lab-Radar Run

- OpenAlex author id(s): {', '.join(meta.get('authors', []))}
- year range: {meta.get('year_from')}–{meta.get('year_to')}
- unique works: {meta.get('unique_count')}
- source: `lab_radar/{slug}/works.jsonl`, `metrics.json`, `report.md`
- disambiguation: automatic top OpenAlex candidate; inspect `lab_radar_resolve_*.json` before publication.

## Topic Change

### Emerging

{chr(10).join(f"- {x['topic']}: {x['early_share']:.1%} → {x['late_share']:.1%} ({x['early']}→{x['late']} papers)" for x in emerging) or "- n/a"}

### Declining

{chr(10).join(f"- {x['topic']}: {x['early_share']:.1%} → {x['late_share']:.1%} ({x['early']}→{x['late']} papers)" for x in declining) or "- n/a"}

## Representative Topics

{chr(10).join(f"- {x['topic']}: {x['papers']} papers" for x in top_topics) or "- n/a"}

## Relation To My Work

- Baldassi/Zecchina/Pittorino 계열은 local entropy, robust ensembles, wide flat minima, typical/atypical solution geometry의 직접 prior art다.
- Zdeborová/Krzakala/Urbani 계열은 perceptron, statistical-to-computational gap, AMP/replica/statistical physics 방향의 확장 context다.
- Chaudhari/Neyshabur/Dziugaite/Jiang 계열은 local entropy or flatness를 optimization/generalization/PAC-Bayes와 연결하는 adjacent line이다.

## Representative Papers

See `lab_radar/{slug}/report.md` notable papers and [[02_Prior_Work_Map]].
"""
        write_md(
            ROOT / "labs" / f"{slug}.md",
            slug.replace("_", " ").title(),
            ["lab-radar", "author-profile"],
            [slug.replace("_", " ").title(), slug],
            f"scholar-lab-radar OpenAlex professor mode: {report.parent}",
            "medium",
            body,
        )


def main_docs() -> None:
    top15 = insertion_paragraphs()
    top20 = pdf_top20()
    index_body = f"""
# Research Radar: Local Entropy DNN Solution Geometry

## Inferred Project

로컬 `Complexity/local_project`의 promoted code/results를 기준으로, 현재 연구는 **Franz-Parisi/local-entropy 계열의 reference-centered shell free-energy/partition-function 측정법을 이론 perceptron에서 검증하고, 이를 3NN 및 MNIST rule-family DNN reference pools에 적용하여 dataset/rule complexity와 solution-space local support의 관계를 진단**하는 프로젝트로 확정했다.

## 핵심 결론 10개

1. local entropy는 spin-glass/perceptron의 constrained-overlap free energy에서 출발해 dense solution cluster를 찾는 도구로 발전했다 [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy].
2. Baldassi-Zecchina 계열은 rare but dense/wide regions가 학습 가능성과 robustness에 중요할 수 있음을 보였다 [@Baldassi2015SubdominantDense; @Baldassi2016UnreasonableEffectiveness; @Baldassi2020ShapingLandscape].
3. 사용자의 theory arm은 analytic full-RS perceptron curve와 shell PM-SAIS sampling curve를 비교하므로, empirical DNN 측정의 estimator validation layer로 쓰기 좋다 [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics].
4. DNN/MNIST arm은 local entropy를 학습 알고리즘으로 쓰기보다, fixed reference 주변 `phi(d)` profile을 QC-aware sampling으로 측정한다는 점이 Entropy-SGD/SAM과 다르다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware].
5. MNIST rule families에서 random label이 가장 큰 `-phi` magnitude를 보이고 structured rules가 낮게 묶인 로컬 결과는 dataset label complexity axis와 연결된다 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal].
6. flatness/generalization을 직접 인과 주장하는 것은 위험하다. reparameterization, symmetry, modern sharpness correlation failures가 강한 반례다 [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids; @Andriushchenko2023ModernSharpness].
7. mode connectivity 문헌은 minima가 고립된 basin이라는 단순 그림을 약화시키므로, 사용자 결과는 “global disconnected basin”보다 “reference-local radial density/support”로 표현해야 한다 [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers].
8. 가장 가까운 최신 인접 prior art는 density-of-states/Wang-Landau식 global loss-spectrum 측정이며, 사용자 novelty는 global DoS가 아니라 reference-local shell curve와 MNIST rule-family axis다 [@Mele2025DensityStates].
9. sampling method 측면에서는 AIS/SMC/vMF 근거가 충분하지만, estimator QC와 split diagnostics를 본문에서 명시해야 한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF].
10. 현재 MNIST 90-ref run은 complete이지만 QC diagnostic pass가 일부 반경에 한정되므로, 본문에서는 “diagnostic evidence”와 “promotion-ready claim”을 구분해야 한다.

## 다음 액션

- Introduction에서는 local entropy의 계보와 “왜 reference-local measurement가 필요한가”를 먼저 세운다.
- Related Work에서는 local entropy / flatness debate / mode connectivity / dataset complexity / sampling estimator를 분리해서 쓴다.
- Experiment에서는 theory validation과 MNIST diagnostic scope를 분리해 QC 기준을 전면에 둔다.
- Discussion에서는 generalization 인과가 아니라 “dataset/rule complexity와 local support geometry의 관찰적 연결”로 제한한다.

## 논문 본문에 바로 넣을 추천 문단 TOP 15

{top15}

## 반드시 읽어야 할 PDF TOP 20

{top20}

## Vault Map

- [[01_Research_Landscape]]
- [[02_Prior_Work_Map]]
- [[03_Novelty_Assessment]]
- [[04_Implications]]
- [[05_Claim_Evidence_Matrix]]
- [[06_Strongest_Prior_Art]]
- [[07_Bibliography.bib]]
"""
    write_md(ROOT / "00_Index.md", "Research Radar Index", ["index", "research-radar"], ["local entropy DNN radar"], "local code/results + megasearch + lab-radar", "high", index_body)

    landscape = """
# 01 Research Landscape

## 현재 연구지형

local entropy 연구는 세 갈래로 나뉜다. 첫째, Franz-Parisi potential과 perceptron/CSP 해공간의 dense cluster 계열은 reference 주변 constrained free energy를 통해 rare but accessible solution regions를 해석한다 [@Franz1995RecipesMetastable; @Baldassi2015SubdominantDense; @Baldassi2016LocalEntropy]. 둘째, deep learning에서는 Entropy-SGD, SAM, SWA처럼 neighborhood geometry를 학습 알고리즘이나 regularizer로 쓰는 흐름이 있다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware; @Izmailov2018AveragingWeights]. 셋째, sharpness/flatness 자체가 generalization explanation으로 충분한지에 대한 강한 회의가 존재한다 [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization; @Andriushchenko2023ModernSharpness].

## 연도별 흐름

| 기간 | 흐름 | 핵심 레퍼런스 |
| --- | --- | --- |
| 1988–1995 | perceptron capacity, version space, Franz-Parisi potential | [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics; @Franz1995RecipesMetastable] |
| 2015–2017 | local entropy/dense clusters and Entropy-SGD | [@Baldassi2015SubdominantDense; @Baldassi2016LocalEntropy; @Chaudhari2017EntropySGD] |
| 2017–2020 | random labels, flatness critique, mode connectivity, generalization measures | [@Zhang2017RethinkingGeneralization; @Dinh2017SharpMinima; @Garipov2018LossSurfaces; @Jiang2020FantasticGeneralization] |
| 2020–2023 | wide flat minima structure, symmetry-aware geometry, SAM/PAC-Bayes debate | [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure; @Pittorino2022DeepNetworksToroids; @Foret2021SharpnessAware] |
| 2024–2026 | atypical perceptron solutions, density-of-states, renewed statistical-physics framing | [@Barbier2024AtypicalSolutions; @Mele2025DensityStates; @Winer2026DeepNeuralNetsHamiltonians] |

## 주요 클러스터

- **Statistical physics local entropy**: Franz-Parisi, perceptron, robust ensembles, atypical high-margin clusters.
- **Optimization toward wide regions**: Entropy-SGD, entropic gradient descent, SAM, SWA.
- **Measurement and geometry**: loss landscape visualization, mode connectivity, density of states.
- **Dataset/rule complexity**: random labels, memorization, graph total variation / label smoothness.
- **Estimator technology**: AIS/SMC/vMF shell sampling and log normalizer estimation.

## Saturation / 전환

flatness-generalization의 단순 상관 주장은 이미 포화와 반례가 많다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness]. 반면 **어떤 coordinate/metric/reference/dataset 조건에서 어떤 local geometry가 관찰되는가**는 여전히 열려 있으며, density-of-states나 reference-local shell profile 같은 측정형 연구가 새롭게 부상 중이다 [@Mele2025DensityStates].
"""
    write_md(ROOT / "01_Research_Landscape.md", "Research Landscape", ["landscape", "literature"], ["research landscape"], "curated verified references + megasearch corpus", "high", landscape)

    prior = """
# 02 Prior Work Map

## 계보

Franz-Parisi constrained potential → perceptron/version-space statistical mechanics → local entropy/dense clusters → Entropy-SGD/entropic algorithms → wide-flat-minima structure → symmetry/flatness critique → density-of-states and dataset-structure measurements.

## A → B → C Evolution

- **Problem evolution**: storage capacity and version space [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics] → metastable/local free energy [@Franz1995RecipesMetastable] → dense solution clusters [@Baldassi2015SubdominantDense] → DNN wide minima and learning landscape [@Baldassi2020ShapingLandscape].
- **Method evolution**: replica/large-deviation theory → local entropy Monte Carlo [@Baldassi2016LocalEntropy] → Entropy-SGD [@Chaudhari2017EntropySGD] → SMC/vMF shell estimator in this project [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF].
- **Data evolution**: random pattern perceptron → true/noisy/random labels [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization] → graph-TV/NMSTV rule families in the local MNIST pipeline [@Shuman2013EmergingGraphSignal; @Ortega2018GraphSignalProcessing].

## 본문에 넣을 레퍼런스 목록

| 논점 | 넣을 레퍼런스 |
| --- | --- |
| local entropy의 원형 | [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy] |
| dense/wide solution regions | [@Baldassi2015SubdominantDense; @Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure] |
| optimization과 구분 | [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware; @Izmailov2018AveragingWeights] |
| flatness caveat | [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids; @Andriushchenko2023ModernSharpness] |
| global landscape/connectivity | [@Li2018VisualizingLoss; @Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers] |
| dataset complexity/control | [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal] |
| sampling/logZ estimator | [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF] |
| recent adjacent threat | [@Mele2025DensityStates; @Barbier2024AtypicalSolutions; @Winer2026DeepNeuralNetsHamiltonians] |
"""
    write_md(ROOT / "02_Prior_Work_Map.md", "Prior Work Map", ["prior-work", "references"], ["prior work map"], "curated verified references", "high", prior)

    novelty = """
# 03 Novelty Assessment

## Novelty Matrix

| 축 | 내 연구 | 가장 가까운 prior art | novelty 판단 |
| --- | --- | --- | --- |
| 문제정의 | reference-centered `phi(d)`/local support를 dataset/rule complexity와 연결 | dense cluster/wide minima theory [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure] | **중-강**: DNN/MNIST rule-complexity axis가 다름 |
| 방법 | analytic perceptron validation + vMF/PM-SAIS shell estimator + QC diagnostics | local entropy Monte Carlo [@Baldassi2016LocalEntropy], AIS/SMC [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers] | **중**: estimator 조합/검증 workflow가 기여 |
| 데이터 | 3NN synthetic grid + MNIST binary rule families(low-TV/even-odd/teacher/random) | random-pattern perceptron, random labels [@Zhang2017RethinkingGeneralization] | **강**: local entropy profile을 rule complexity ladder에 직접 매핑 |
| 이론 | Full-RS perceptron baseline과 sampler convergence 비교 | perceptron statistical mechanics [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics] | **중**: 새 이론보다 validation bridge |
| 평가 | QC pass, split diagnostics, bootstrap/ESS, reference-pool stability | generalization measure studies [@Jiang2020FantasticGeneralization] | **중-강**: measurement hygiene를 전면화 |
| 적용맥락 | MNIST reference family/local support diagnostic | density of states [@Mele2025DensityStates] | **중**: global DoS와 달리 reference-local profile |
| 한계극복 | flatness를 generalization causal claim이 아닌 diagnostic quantity로 제한 | sharpness critiques [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness] | **강한 방어 논리** |

## 강한 novelty 후보

1. **이론-실험 bridge**: analytic perceptron local entropy와 shell estimator를 먼저 비교한 뒤 DNN/MNIST로 이식.
2. **dataset/rule complexity axis**: NMSTV/graph-TV 기반 rule ordering과 `phi(d)` profile을 연결.
3. **QC-aware local entropy measurement**: split/ESS/bootstrap diagnostics를 claim boundary로 사용.
4. **reference-family analysis**: single optimum이 아니라 trained references의 distribution/cluster를 분석.

## 약한 novelty 후보

- “flat minima가 generalization을 설명한다”는 주장은 기존 논쟁이 강하므로 novelty로 쓰면 위험하다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness].
- “local entropy를 최적화한다”는 주장은 Entropy-SGD/entropic gradient prior와 겹친다 [@Chaudhari2017EntropySGD; @Baldassi2021EntropicGradient].
"""
    write_md(ROOT / "03_Novelty_Assessment.md", "Novelty Assessment", ["novelty", "assessment"], ["novelty matrix"], "local project evidence + curated literature", "high", novelty)

    implications = """
# 04 Implications

## 주장 가능한 문장

- 본 연구는 local entropy를 generalization의 단일 원인으로 주장하기보다, fixed reference 주변의 radial support/free-energy profile을 측정하는 진단 도구로 사용한다 [@Dinh2017SharpMinima; @Jiang2020FantasticGeneralization; @Andriushchenko2023ModernSharpness].
- local entropy/dense cluster 계열은 simple neural models에서 rare but wide regions가 알고리즘적으로 중요할 수 있음을 보여 왔고, 본 연구는 그 관점을 DNN/MNIST reference-family measurement로 확장한다 [@Baldassi2015SubdominantDense; @Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure].
- random-label 및 structured-label controls는 dataset complexity가 training behavior와 solution geometry를 바꿀 수 있음을 검토하는 자연스러운 실험축이다 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal].

## 주장하면 안 되는 문장

- “local entropy가 높으면 항상 generalization이 좋다.” → reparameterization, data dependence, modern sharpness counter-evidence 때문에 과장이다 [@Dinh2017SharpMinima; @Andriushchenko2023ModernSharpness].
- “MNIST 90-ref 결과는 모든 반경에서 QC-validated이다.” → local report상 diagnostic QC pass가 제한적이므로 불가.
- “global loss landscape의 모든 density of states를 측정했다.” → 본 연구는 reference-centered shell profile이며 global DoS와 다르다 [@Mele2025DensityStates].

## 학술적/방법론적 함의

- 학술적으로는 local entropy theory를 empirical DNN diagnostic protocol로 번역한다.
- 방법론적으로는 logZ/partition-function estimator와 QC criteria를 함께 제시해 후속 연구가 재현 가능한 local geometry measurement를 할 수 있게 한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers].
- 실무적으로는 dataset/rule complexity가 trained solution support profile에 미치는 영향을 진단하는 lightweight benchmark를 제안할 수 있다.
"""
    write_md(ROOT / "04_Implications.md", "Implications", ["implications", "claims"], ["claim boundaries"], "curated literature + local reports", "high", implications)

    matrix = """
# 05 Claim Evidence Matrix

| claim | evidence | refs | confidence | insertion_point | basis |
| --- | --- | --- | --- | --- | --- |
| local entropy는 reference 주변 solution density/free energy를 측정하는 통계물리 quantity다 | Franz-Parisi potential과 CSP local entropy 계보 | [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy] | high | Introduction | metadata+abstract/PDF |
| dense/wide solution regions는 simple neural models에서 학습 가능성과 연결된다 | discrete synapse/perceptron/WFM 연구 | [@Baldassi2015SubdominantDense; @Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure] | high | Related Work | metadata+abstract/PDF |
| 본 연구는 optimization method가 아니라 measurement protocol이다 | Entropy-SGD/SAM은 학습 objective, 본 연구는 fixed-reference phi(d) 측정 | [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware] | high | Method | metadata+abstract/PDF+local code |
| raw L2-shell flatness는 reparameterization/symmetry caveat를 가진다 | sharpness critique and toroid/symmetry literature | [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids] | high | Limitation | metadata+abstract/PDF |
| dataset/rule complexity controls가 필요하다 | random labels/memorization and graph signal smoothness | [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization; @Shuman2013EmergingGraphSignal] | high | Experiment | metadata+abstract/PDF |
| SMC/vMF shell estimator는 logZ estimation foundation 위에 있다 | AIS/SMC/vMF sampling literature | [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers; @Wood1994SimulationVMF] | high | Method | metadata+PDF |
| density-of-states는 가장 가까운 adjacent measurement prior다 | Wang-Landau global DoS for NN loss spectrum | [@Mele2025DensityStates] | high | Related Work | OpenReview+arXiv |
| MNIST 결과는 현 단계에서 diagnostic claim이 적합하다 | local reports: 9000/9000 complete, QC pass 5/100 in mechanical 90ref, suitability limited | local project reports | high | Limitation | local report/code |

## 본문 삽입 제안

""" + insertion_sections()
    write_md(ROOT / "05_Claim_Evidence_Matrix.md", "Claim Evidence Matrix", ["claims", "evidence"], ["claim evidence matrix"], "curated literature + local project reports", "high", matrix)

    threats = """
# 06 Strongest Prior Art

| prior art | 왜 위협적인가 | 방어 논리 |
| --- | --- | --- |
| [@Baldassi2016LocalEntropy] | local entropy as sampling objective를 이미 제시 | 본 연구는 CSP solver가 아니라 DNN/MNIST reference-local shell profile + QC validation |
| [@Chaudhari2017EntropySGD] | local entropy를 DNN optimization에 적용 | 본 연구는 optimization이 아니라 measurement and diagnostics |
| [@Baldassi2020ShapingLandscape] | wide flat minima in neural networks와 가장 가까움 | simple model theory 중심; 본 연구는 rule complexity axis와 empirical reference-pool estimator |
| [@Baldassi2021UnveilingStructure] | wide flat minima structure 자체를 깊게 분석 | 사용자 결과는 그 구조를 empirical `phi(d)` profile로 관찰/검정하는 방향 |
| [@Mele2025DensityStates] | dataset structure와 parameter-space density를 직접 연결 | global density-of-states vs reference-centered radial local entropy; estimator와 claim이 다름 |
| [@Dinh2017SharpMinima] | flatness 측정의 coordinate dependence를 공격 | limitation으로 수용하고, fixed architecture/regularized coordinate/QC diagnostic로 claim 범위 제한 |
| [@Andriushchenko2023ModernSharpness] | modern setting에서 sharpness-generalization 상관을 약화 | generalization causality 대신 local geometry measurement로 framing |
| [@Pittorino2022DeepNetworksToroids] | symmetry 제거 필요성을 제기 | raw L2 distance caveat를 명시하고 future work로 quotient/invariant distance 제안 |
| [@Zhang2017RethinkingGeneralization] | random labels에서도 fit 가능함을 보여 단순 capacity/flatness 설명을 약화 | random_label을 control로 포함하여 local support profile 차이를 직접 측정 |
| [@Garipov2018LossSurfaces] | minima가 low-loss paths로 연결될 수 있음 | global connectivity와 reference-local radial density는 상보적 quantity라고 방어 |
"""
    write_md(ROOT / "06_Strongest_Prior_Art.md", "Strongest Prior Art", ["prior-art", "threats"], ["strongest prior art"], "curated references", "high", threats)

    unverified = """
# TODO Unverified Citations

다음 항목은 흥미롭지만 DOI/arXiv/venue/year 확인이 충분하지 않거나 현재 corpus에서 source corroboration이 약해 본문 추천 인용에는 쓰지 않았다.

| candidate | reason | action |
| --- | --- | --- |
| CVPR 2026 Globscope | future/current venue page from web search; metadata may change | official proceedings once available |
| assorted blog posts on sharpness/loss landscapes | non-peer-reviewed | use only for orientation |
| some Crossref/OpenAlex broad survey hits in `megasearch/corpus.json` | query drift from broad terms | exclude unless manually verified |
| Semantic Scholar failed facets | 429 rate limit | rerun later with API key or lower rate |
"""
    write_md(ROOT / "99_TODO_Unverified_Citations.md", "TODO Unverified Citations", ["todo", "unverified"], ["unverified citations"], "megasearch manifest", "high", unverified)


def insertion_paragraphs() -> str:
    sections = insertion_data()
    lines = []
    for i, item in enumerate(sections[:15], 1):
        lines.append(f"{i}. **{item['position']}**: {item['text']}")
    return "\n".join(lines)


def insertion_sections() -> str:
    lines = []
    for item in insertion_data():
        lines.append(
            f"""
### {item['title']}

- 넣을 위치: {item['position']}
- 본문에 넣을 내용: {item['text']}
- 근거: {item['evidence']}
- 주의: {item['caution']}
"""
        )
    return "\n".join(lines)


def insertion_data() -> list[dict[str, str]]:
    return [
        {
            "title": "Local Entropy Framing",
            "position": "Introduction",
            "text": "본 연구는 학습된 한 해 주변의 단순한 손실값이 아니라, reference parameter로부터의 거리 \\(d\\)에서 유지되는 해공간의 유효 부피를 측정 대상으로 삼는다. 이러한 관점은 고정된 reference와의 overlap을 제한한 자유에너지로 metastable structure를 해석하는 Franz-Parisi potential 및 local entropy 계보와 맞닿아 있다 [@Franz1995RecipesMetastable; @Baldassi2016LocalEntropy].",
            "evidence": "Franz-Parisi potential과 CSP local entropy의 원형 논문 2개.",
            "caution": "spin-glass/CSP에서 DNN으로의 직접 등식이 아니라 방법론적 계보로만 쓴다.",
        },
        {
            "title": "Dense Cluster Motivation",
            "position": "Introduction",
            "text": "선행연구는 신경망의 해공간이 단순히 많은 isolated minima로만 구성되는 것이 아니라, 드물지만 조밀하고 접근 가능한 영역을 포함할 수 있음을 보여 왔다 [@Baldassi2015SubdominantDense; @Baldassi2016UnreasonableEffectiveness]. 따라서 특정 reference 주변의 local support profile을 직접 측정하는 것은 wide/robust solution hypothesis를 empirical하게 점검하는 한 방법이 된다.",
            "evidence": "dense cluster와 robust ensemble의 seminal papers.",
            "caution": "‘좋은 일반화’를 바로 결론내리지 않는다.",
        },
        {
            "title": "Wide Flat Minima Prior",
            "position": "Related Work",
            "text": "wide flat minima는 simple neural-network models에서 높은 margin 중심부와 그 주변의 dense solution structure로 해석되어 왔다 [@Baldassi2020ShapingLandscape; @Baldassi2021UnveilingStructure]. 본 연구는 이러한 이론적 그림을 전제로 삼되, 학습 알고리즘을 새로 제안하기보다 reference pool 주변의 \\(\\phi(d)\\) 곡선을 추정하는 측정 문제로 재구성한다.",
            "evidence": "PNAS 2020, PRL 2021.",
            "caution": "‘전제로 삼되 검증한다’는 표현을 유지.",
        },
        {
            "title": "Optimization Distinction",
            "position": "Related Work",
            "text": "Entropy-SGD와 SAM은 parameter neighborhood 정보를 학습 objective에 넣어 wide 혹은 sharpness-aware solution을 찾는 optimization 계열이다 [@Chaudhari2017EntropySGD; @Foret2021SharpnessAware]. 반면 본 연구의 중심은 optimization 성능 개선이 아니라, 이미 얻어진 reference solutions 주변에서 shell-wise partition estimate와 QC diagnostics를 통해 local geometry를 측정하는 것이다.",
            "evidence": "Entropy-SGD and SAM.",
            "caution": "방법 비교에서 성능 우열을 주장하지 않는다.",
        },
        {
            "title": "Sampling Foundation",
            "position": "Method",
            "text": "거리 shell에서의 partition-function 추정은 normalizing constant estimation 문제로 볼 수 있으며, annealed importance sampling과 SMC samplers는 이러한 분포열 기반 추정의 표준적 근거를 제공한다 [@Neal2001AnnealedImportance; @DelMoral2006SMCSamplers]. 본 구현은 hypersphere 방향 샘플링을 위해 von Mises-Fisher proposal을 사용하며, 이는 directional statistics의 표준 sampling scheme에 근거한다 [@Wood1994SimulationVMF].",
            "evidence": "AIS, SMC, vMF sampling.",
            "caution": "구현 세부는 local code와 QC criteria로 별도 설명.",
        },
        {
            "title": "Theory Validation",
            "position": "Method",
            "text": "이론 검증 단계에서는 perceptron local-entropy curve를 analytic full-RS baseline과 shell sampling estimate로 동시에 계산하여 estimator의 방향성을 점검한다. 이러한 설계는 perceptron solution-space를 통계물리적으로 분석한 고전 연구와 Franz-Parisi 계보를 DNN 측정으로 옮기기 전의 calibration layer로 기능한다 [@Gardner1988SpaceInteractions; @Seung1992StatisticalMechanics; @Franz1995RecipesMetastable].",
            "evidence": "perceptron/statistical mechanics lineage and local project theory comparison.",
            "caution": "새로운 closed-form theory claim으로 쓰지 않는다.",
        },
        {
            "title": "Dataset Complexity Axis",
            "position": "Experiment",
            "text": "MNIST rule-family 실험은 true-structured, teacher-generated, low-TV, random-label 조건을 함께 두어 label complexity가 local support geometry에 미치는 영향을 관찰하도록 설계되었다. random labels가 capacity와 memorization 문제를 드러내는 강한 control임은 기존 연구에서 반복적으로 확인되었고 [@Zhang2017RethinkingGeneralization; @Arpit2017CloserMemorization], graph total variation은 data geometry 위 label smoothness를 표현하는 자연스러운 축이다 [@Shuman2013EmergingGraphSignal; @Ortega2018GraphSignalProcessing].",
            "evidence": "random label/memorization and graph signal processing references.",
            "caution": "n=4 rule correlation은 exploratory로 제한.",
        },
        {
            "title": "Flatness Caveat",
            "position": "Limitation",
            "text": "parameter-space flatness는 reparameterization과 symmetry에 민감하므로, 본 연구의 L2-shell profile 역시 특정 architecture, regularization, coordinate convention 아래의 diagnostic quantity로 해석해야 한다 [@Dinh2017SharpMinima; @Pittorino2022DeepNetworksToroids]. 따라서 본문에서는 \\(\\phi(d)\\)를 universal generalization measure가 아니라 fixed protocol에서의 reference-local support profile로 부른다.",
            "evidence": "Dinh critique and toroid/symmetry paper.",
            "caution": "generalization 인과 주장 금지.",
        },
        {
            "title": "Modern Sharpness Debate",
            "position": "Discussion",
            "text": "최근의 대규모 sharpness 연구는 sharpness와 generalization의 관계가 architecture, hyperparameter, data setting에 따라 일관되지 않을 수 있음을 보였다 [@Andriushchenko2023ModernSharpness]. 본 연구의 기여는 이 논쟁을 우회하여, generalization을 직접 예측하기보다 dataset/rule condition에 따른 local support geometry의 변화를 측정하는 데 있다.",
            "evidence": "modern sharpness empirical critique.",
            "caution": "우회한다는 표현은 논쟁을 무시한다는 뜻이 아니라 claim을 제한한다는 뜻.",
        },
        {
            "title": "Mode Connectivity Caveat",
            "position": "Discussion",
            "text": "mode connectivity 연구는 독립적으로 학습된 solutions가 low-loss curves로 연결될 수 있음을 보여, minima를 고립된 basin으로 보는 단순 그림을 약화시킨다 [@Garipov2018LossSurfaces; @Draxler2018EssentiallyNoBarriers]. 따라서 본 연구의 radial shell profile은 global connectivity의 대체물이 아니라, 특정 reference 주변의 local density/support를 보는 상보적 측정으로 해석된다.",
            "evidence": "mode connectivity papers.",
            "caution": "local profile에서 global topology를 추론하지 않는다.",
        },
        {
            "title": "Density Of States Threat",
            "position": "Related Work",
            "text": "최근에는 Wang-Landau sampling으로 neural network의 global density of states를 추정하고 dataset structure와 loss spectrum의 관계를 분석하는 연구도 등장했다 [@Mele2025DensityStates]. 본 연구는 이와 달리 전체 loss spectrum이 아니라 trained reference 주변의 거리별 shell profile을 추정하므로, global DoS와 reference-local local entropy를 구분해 비교한다.",
            "evidence": "TMLR 2025 density-of-states paper.",
            "caution": "가장 가까운 adjacent prior로 정직하게 다룬다.",
        },
        {
            "title": "Atypical Solutions",
            "position": "Related Work",
            "text": "atypical high-margin solutions와 그 주변 local entropy는 binary/symmetric perceptron에서 최근까지 활발히 분석되고 있다 [@Baldassi2023TypicalAtypical; @Barbier2024AtypicalSolutions]. 이러한 결과는 효율적 알고리즘이 exponentially dominant typical solutions가 아니라 rare structured regions를 찾을 수 있다는 해석을 뒷받침한다.",
            "evidence": "PRE 2023 and J Phys A/arXiv 2024.",
            "caution": "MNIST DNN 결과와 동일한 phase transition이라고 쓰지 않는다.",
        },
        {
            "title": "PAC-Bayes Link",
            "position": "Discussion",
            "text": "flat minima와 generalization의 formal link는 PAC-Bayes 관점에서도 연구되어 왔으며, 최근에는 gradient 및 functional inequality를 이용해 dimension-explicit dependence를 줄이는 bound가 제안되었다 [@Dziugaite2017ComputingNonvacuous; @Haddouche2025PACBayesianLink]. 다만 본 연구의 shell entropy는 bound 자체가 아니라 empirical diagnostic이므로, PAC-Bayes 연결은 해석적 가능성으로만 제시한다.",
            "evidence": "PAC-Bayes flatness references.",
            "caution": "PAC-Bayes bound를 계산했다고 오해하게 쓰지 않는다.",
        },
        {
            "title": "Local Results Scope",
            "position": "Experiment",
            "text": "로컬 MNIST 결과는 90 references per rule 및 25개 radius grid의 mechanical sampling을 완료했지만, diagnostic QC pass는 일부 rule-radius에 제한되어 있다. 따라서 본문에서는 full grid를 exploratory measurement로, QC-passed subset을 stronger evidence로 구분해 보고한다.",
            "evidence": "local `03_dnn_mnist/04_sampling/raw_outputs/refpool1024_all_radii_90ref/REPORT.md` and final_goal_report.",
            "caution": "이 문장에는 문헌 인용보다 local report 경로를 footnote/appendix로 붙인다.",
        },
        {
            "title": "Main Contribution",
            "position": "Introduction",
            "text": "요약하면, 본 연구의 기여는 local entropy를 학습 알고리즘이나 보편적 일반화 지표로 주장하는 것이 아니라, theory-validated shell estimator와 rule-complexity controls를 결합하여 DNN reference 주변 solution support를 측정하는 재현 가능한 protocol을 제시하는 데 있다 [@Baldassi2016LocalEntropy; @Neal2001AnnealedImportance; @Mele2025DensityStates].",
            "evidence": "local entropy, sampling, and adjacent DoS references.",
            "caution": "‘보편적’ claim을 피하고 protocol contribution을 강조.",
        },
    ]


def pdf_top20_keys() -> list[str]:
    return [
        "Baldassi2016LocalEntropy",
        "Baldassi2015SubdominantDense",
        "Baldassi2016UnreasonableEffectiveness",
        "Chaudhari2017EntropySGD",
        "Baldassi2020ShapingLandscape",
        "Baldassi2021UnveilingStructure",
        "Pittorino2022DeepNetworksToroids",
        "Baldassi2022LearningAtypical",
        "Baldassi2023TypicalAtypical",
        "Barbier2024AtypicalSolutions",
        "Mele2025DensityStates",
        "Dinh2017SharpMinima",
        "Andriushchenko2023ModernSharpness",
        "Jiang2020FantasticGeneralization",
        "Zhang2017RethinkingGeneralization",
        "Arpit2017CloserMemorization",
        "Garipov2018LossSurfaces",
        "Neal2001AnnealedImportance",
        "DelMoral2006SMCSamplers",
        "Foret2021SharpnessAware",
    ]


def pdf_top20() -> str:
    selected = pdf_top20_keys()
    by_key = {r["key"]: r for r in REFS}
    lines = []
    for i, key in enumerate(selected, 1):
        r = by_key[key]
        pdf = r.get("pdf") or r.get("url")
        lines.append(f"{i}. [@{key}] {r['title']} ({r['year']}) — {pdf}")
    return "\n".join(lines)


def maps() -> None:
    map_specs = {
        "statistical_physics_local_entropy": [
            "Franz1995RecipesMetastable",
            "Gardner1988SpaceInteractions",
            "Seung1992StatisticalMechanics",
            "Baldassi2015SubdominantDense",
            "Baldassi2016LocalEntropy",
            "Baldassi2016UnreasonableEffectiveness",
            "Baldassi2020ShapingLandscape",
            "Baldassi2021UnveilingStructure",
            "Baldassi2022LearningAtypical",
            "Baldassi2023TypicalAtypical",
            "Barbier2024AtypicalSolutions",
        ],
        "flatness_generalization_debate": [
            "Keskar2017LargeBatch",
            "Dinh2017SharpMinima",
            "Jiang2020FantasticGeneralization",
            "Andriushchenko2023ModernSharpness",
            "Foret2021SharpnessAware",
            "Dziugaite2017ComputingNonvacuous",
            "Haddouche2025PACBayesianLink",
        ],
        "loss_landscape_connectivity": [
            "Li2018VisualizingLoss",
            "Garipov2018LossSurfaces",
            "Draxler2018EssentiallyNoBarriers",
            "Izmailov2018AveragingWeights",
            "Pittorino2022DeepNetworksToroids",
        ],
        "dataset_complexity_memorization": [
            "Zhang2017RethinkingGeneralization",
            "Arpit2017CloserMemorization",
            "Shuman2013EmergingGraphSignal",
            "Ortega2018GraphSignalProcessing",
            "Zhou2004LearningLocalGlobal",
            "Mele2025DensityStates",
        ],
        "sampling_partition_estimators": [
            "Neal2001AnnealedImportance",
            "DelMoral2006SMCSamplers",
            "Wood1994SimulationVMF",
            "Baldassi2016LocalEntropy",
            "Mele2025DensityStates",
        ],
    }
    by_key = {r["key"]: r for r in REFS}
    for name, keys in map_specs.items():
        body = f"# {name.replace('_', ' ').title()}\n\n"
        body += "## Papers\n\n"
        for key in keys:
            body += f"- [[papers/{key}|{key}]]: {by_key[key]['cite_for']}\n"
        body += "\n## Connections\n\n"
        body += "- " + " → ".join(f"[[papers/{k}|{k}]]" for k in keys[:5]) + "\n"
        body += "- Back to [[00_Index]] and [[02_Prior_Work_Map]].\n"
        write_md(ROOT / "maps" / f"{name}.md", name, ["map", "topic-cluster"], [name.replace("_", " ")], "curated references", "high", body)


def megasearch_summary() -> None:
    manifest_path = ROOT / "megasearch" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status_counts = {}
        for m in manifest:
            status_counts[m.get("status", "unknown")] = status_counts.get(m.get("status", "unknown"), 0) + 1
    else:
        status_counts = {}
    corpus_count = 0
    shortlist_count = 0
    if (ROOT / "megasearch" / "corpus.json").exists():
        corpus_count = len(json.loads((ROOT / "megasearch" / "corpus.json").read_text(encoding="utf-8")))
    if (ROOT / "megasearch" / "corpus_shortlist_min2.json").exists():
        shortlist_count = len(json.loads((ROOT / "megasearch" / "corpus_shortlist_min2.json").read_text(encoding="utf-8")))
    summary = f"""
# Megasearch Summary

- depth: L5 Total/exhaustive fallback
- topic: local entropy, Franz-Parisi potential, shell sampling, neural-network solution geometry, dataset complexity
- raw source count status: {status_counts}
- unique corpus: {corpus_count}
- min-source>=2 shortlist: {shortlist_count}
- source note: scholar-megasearch was cloned and its merge/fetch scripts were used. Full MCP fan-out was unavailable in this session, so raw acquisition used local fallback APIs: arXiv, OpenAlex, Crossref, DBLP, DDG, PubMed, and partial Semantic Scholar. Semantic Scholar rate-limited several facets with HTTP 429; this is recorded in `manifest.json`.

## Query Facets

See `query_plan.json`.

## Recommended Use

Use `corpus_shortlist_min2.*` as high-precision discovery evidence, but use `07_Bibliography.bib` for manuscript citations because it has been manually curated for DOI/arXiv/venue/year hygiene.
"""
    (ROOT / "megasearch" / "summary.md").write_text(summary.strip() + "\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    generate_bib()
    paper_cards()
    lab_cards()
    main_docs()
    maps()
    megasearch_summary()


if __name__ == "__main__":
    main()
