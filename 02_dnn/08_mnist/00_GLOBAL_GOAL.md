# 00_GLOBAL_GOAL — MNIST14 PM-SAIS research contract

## One-sentence goal

\[
\boxed{
\text{Use one real MNIST input marginal, vary only the label rule, and compare reference-conditioned shell free entropy } \phi(d).
}
\]

## Why this experiment exists

The previous 2D synthetic pipeline established a PM-SAIS/RLI route for finite 3NN shell entropy. The MNIST14 extension is a closure experiment:

- input is now real image data, not synthetic 2D fields;
- morphology is preserved by 14x14 average pooling;
- architecture is redesigned but kept small enough for shell sampling;
- the estimator is fixed before looking at results.

## Main scientific design

### Data

\[
28\times28 \rightarrow 14\times14 \rightarrow x\in\mathbb R^{196}.
\]

Use exact 2x2 average pooling. Store both raw14 and standardized vectors.

### Label rules

All regimes share the same input marginal.

| rule | definition | interpretation |
|---|---|---|
| `real_even_odd` | \(y=+1\) for even digit, \(-1\) for odd digit | semantic structured label |
| `teacher_nn` | \(y=\operatorname{sign}(T(x)-\operatorname{median}_{train}T)\) | architecture-compatible synthetic rule |
| `random_label` | iid balanced random \(y\in\{-1,+1\}\) | no label structure / memorization control |

### Model

Main architecture:

\[
196\rightarrow16\rightarrow16\rightarrow1,\quad \tanh.
\]

Parameter count:

\[
P=196\cdot16+16+16\cdot16+16+16+1=3441.
\]

Backup architecture only if random-label exact reference rate fails badly:

\[
196\rightarrow24\rightarrow24\rightarrow1,\quad P=5353.
\]

### Loss convention

Labels \(y_i\in\{-1,+1\}\), logit \(f_\theta(x_i)\).

\[
\ell_i(\theta)=\log(1+\exp[-y_i f_\theta(x_i)]).
\]

\[
CE_{\rm mean}=\frac1n\sum_i\ell_i,\qquad CE_{\rm sum}=n\,CE_{\rm mean}.
\]

Sampling target:

\[
U(\theta)=\beta CE_{\rm sum}(\theta)+\lambda_{\rm reg}\frac{\|\theta\|^2}{2P}.
\]

If code returns `CE_mean`, use:

\[
\gamma_{\rm CE}=\beta n_{\rm train}
\]

and compute residual weights with \(\exp[-\gamma_{\rm CE}CE_{\rm mean}]\).

Default:

\[
\beta=1,\qquad \lambda_{\rm reg}\in\{1,10,50,100\}\text{ selected by pilot.}
\]

### Pool 1

Pool 1 is the reference ensemble.

Practical reference law:

\[
\boxed{\text{optimizer-induced exact reference ensemble}}
\]

Acceptance:

\[
\mathrm{train\ error}(\tilde\theta)=0.
\]

Record reference-bias diagnostics:

- `exact_opt_unweighted`
- `exact_opt_L2_reweighted`
- `norm_matched_exact`

Do not claim exact sampling from:

\[
P_{\rm ref}^{0}(\theta\mid D)
\propto
\mathbf 1\{\mathrm{err}=0\}
e^{-\lambda_{\rm ref}\|\theta\|^2/(2P)}
\]

unless a target-aware reference sampler is implemented.

### Pool 2

Pool 2 is the reference-conditioned hard shell.

\[
d=d_{\rm raw}=\frac{\|\theta-\tilde\theta\|}{\sqrt P}.
\]

\[
\theta(d,u)=\tilde\theta+\sqrt P\,d\,u,\qquad \|u\|=1.
\]

Main estimator: PM-SAIS \(H=\infty\). Optional PM-RLI \(H\in\{8,4,2\}\) diagnostic.

### Main observables

\[
Z(d)=e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim{\rm vMF}}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u))}
].
\]

\[
\Delta\phi_{\rm full}(d)
=
\frac{P-1}{P}\log\frac d{d_0}
+
\frac1P[\log Z(d)-\log Z(d_0)].
\]

\[
\Delta\phi_{\rm energy}(d)
=
\frac1P[\log Z(d)-\log Z(d_0)].
\]

Main figure: three curves of \(\Delta\phi_{\rm energy}(d)\) for `real_even_odd`, `teacher_nn`, `random_label`.

## Claim policy

Allowed:

\[
\boxed{
\text{Same input marginal, different label rules, different PM-SAIS shell free-entropy profiles.}
}
\]

Not allowed:

\[
\boxed{
\text{We invented local entropy.}
}
\]

\[
\boxed{
\text{Optimizer-found references are exact }P_{\rm ref}^{0}\text{ samples.}
}
\]

\[
\boxed{
\text{Any radius beyond QC pass is supported.}
}
\]
