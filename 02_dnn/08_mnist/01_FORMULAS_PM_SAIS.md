# 01_FORMULAS_PM_SAIS

## PM-SAIS identity

General shell partition:

\[
\Omega(s)=A(s)\mathbb E_{u\sim p_s}[F_s(u)].
\]

PM-SAIS estimates the same target using a proposal \(q_s\):

\[
\mathbb E_{p_s}[F_s(u)]
=
\mathbb E_{q_s}\left[F_s(u)\frac{p_s(u)}{q_s(u)}\right].
\]

For this project:

\[
s=d=d_{\rm raw},\qquad p_s={\rm Haar}(S^{P-1}).
\]

## Hard shell map

\[
\theta(d,u)=\tilde\theta+\sqrt P\,d\,u,\qquad \|u\|=1.
\]

## Target

\[
U(\theta)=\gamma CE_{\rm mean}(\theta;D)+\lambda\frac{\|\theta\|^2}{2P}.
\]

With extensive CE:

\[
\gamma=\beta n_{\rm train}.
\]

## L2 decomposition

\[
\|\theta(d,u)\|^2
=
\|\tilde\theta\|^2
+
Pd^2
+
2\sqrt P\,d\,\|\tilde\theta\|(\hat{\tilde\theta}\cdot u).
\]

Let:

\[
\mu_d=-\hat{\tilde\theta},\qquad
\kappa_d=\lambda d\frac{\|\tilde\theta\|}{\sqrt P}.
\]

Then the L2 angular tilt is matched by:

\[
q_d(u)={\rm vMF}(\mu_d,\kappa_d).
\]

## PM-SAIS angular partition

Dropping the reference-constant factor that cancels in relative curves:

\[
Z_{\rm PM-SAIS}(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim q_d}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u);D)}
].
\]

## Monte Carlo estimator

\[
\widehat Z(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\frac1m\sum_{a=1}^{m}
\exp[-\gamma CE_{\rm mean}(\theta(d,u_a);D)].
\]

Use logsumexp:

\[
\log\widehat Z(d)
=
-\lambda d^2/2+\log M_P(\kappa_d)
+\operatorname{logmeanexp}_a[-\gamma CE_a].
\]

## PM-RLI optional H-gate

\[
h(\theta)=\sqrt{2\max(CE_{\rm mean}(\theta)-CE_{\rm mean}(\tilde\theta),0)}.
\]

\[
Z_H(d)
=
e^{-\lambda d^2/2}M_P(\kappa_d)
\mathbb E_{u\sim q_d}
[
e^{-\gamma CE_{\rm mean}(\theta(d,u))}
\mathbf 1\{h(\theta(d,u))\le H\}
].
\]

\[
R_H(d)=\frac{Z_H(d)}{Z_\infty(d)}.
\]

Interpretation:

| layer | role |
|---|---|
| \(H=\infty\) | main PM-SAIS/global shell |
| \(H=8\) | broad loss-response diagnostic |
| \(H=4\) | medium sector diagnostic |
| \(H=2\) | strict local low-loss diagnostic only |

## Full and energy-only free entropy

\[
\Delta\phi_{\rm full}(d;H)
=
\frac{P-1}{P}\log\frac d{d_0}
+
\frac1P[\log Z_H(d)-\log Z_H(d_0)].
\]

\[
\Delta\phi_{\rm energy}(d;H)
=
\frac1P[\log Z_H(d)-\log Z_H(d_0)].
\]

Use \(\Delta\phi_{\rm energy}\) as the main landscape-quality comparison. Show \(\Delta\phi_{\rm full}\) with area decomposition.
