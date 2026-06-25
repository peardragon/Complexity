# Final Method Alignment

This benchmark treats vMF + L2 as the final method in proxy coordinates.

## Shared estimator

The estimator is self-normalized importance sampling:

```text
I_f = sum_i w_i f(z_i) / sum_i w_i
w_i = exp(-beta E(z_i)) / q(z_i)
```

In the DNN experiments, `z_i` corresponds to a parameter-space shell sample
around a reference solution, `E` corresponds to the CE-like energy, and `q`
combines L2 shell selection with a vMF direction proposal. In this proxy,
`z_i` is two-dimensional so that the final figure can show the entire
landscape.

## Proposal mapping

| DNN method object | Proxy object |
| --- | --- |
| Reference parameter `theta_ref` | Origin / `solution_core` center |
| L2 distance from reference | Polar radius `r` |
| Direction on high-dimensional sphere | Angle on the unit circle |
| vMF around selected directions | von Mises around region angles |
| CE-like energy | Proxy energy `E(z)` |
| Weighted QC ratios | Self-normalized region mass and ESS |

The implementation should therefore be read as a dimensional reduction of the
final method, not as a different method.

