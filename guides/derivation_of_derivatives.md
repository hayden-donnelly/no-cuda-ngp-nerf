## Preliminaries:
```math
$$
\begin{align}

c =&\ c_0\alpha_0 + c_1\alpha_1(1 - \alpha_0) + ... + 
    c_n\alpha_n(1 - \alpha_0) ... (1 - \alpha_n) \notag\\

d =&\ t_0\alpha_0 + t_1\alpha_1(1 - \alpha_0) + .. + 
    t_n\alpha_n(1 - \alpha_0) ... (1 - \alpha_n) \notag\\

o_p =&\ 1 - (1 - \alpha_0)(1 - \alpha_1) ... (1 - \alpha_n) \notag\\

\notag\\

\alpha_i =&\ 1 - e^{-\delta_i\sigma_i} \text{, so } 
    \frac{\partial\alpha_i}{\partial\sigma_i} = 
    \delta_i - e^{-\delta_i\sigma_i} = 
    \delta_i(1 - \alpha_i)\notag\\

\notag\\

\text{Want:}&\ \frac{\partial L}{\partial\sigma_0} \notag\\

\notag\\

\text{Have:}&\ \frac{\partial L}{\partial c} \text{, } 
    \frac{\partial c}{\partial d} \text{, } \frac{\partial L}{\partial o_p} \notag\\

\notag\\

\text{Chain Rule:}&\ \frac{\partial L}{\partial\sigma_0} = 
    \frac{\partial L}{\partial c} \cdot \frac{\partial c}{\partial\sigma_0} + 
    \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial\sigma_0} +
    \frac{\partial L}{\partial o_p} \cdot \frac{\partial o_p}{\partial\sigma_0} \notag\\

\end{align}
$$
```

## 1.
```math
$$
\begin{align}
\frac{\partial c}{\partial\sigma_0} =&\ 
    \frac{\partial c}{\partial\alpha_0} \cdot 
    \frac{\partial\alpha_0}{\partial\sigma_0} + 
    \sum_{i = 1}^{N}{
        \frac{\partial c}{\partial\alpha_i} \cdot \frac{\partial\alpha_i}{\partial\sigma_0}
    } \notag\\

\frac{\partial c}{\partial\sigma_0} =&\ 
    \frac{\partial c}{\partial\alpha_0} \cdot 
    \frac{\partial\alpha_0}{\partial\sigma_0} + 
    \underbrace{
        \cancel{
            \sum_{i = 1}^{N}{
                \frac{\partial c}{\partial\alpha_i} \cdot 
                \frac{\partial\alpha_i}{\partial\sigma_0}
            }
        }
    }_{\rlap{\text{$= 0$ since $\alpha_i$ doesn't depend on $\sigma_0$.}}} \notag\\

\notag\\

=&\ [c_0 - c_1\alpha_1 - ... - c_n\alpha_n(1 - \alpha_1) ... (1 - \alpha_{n - 1})] 
    \cdot [\delta_0(1 - \alpha_0)] \notag\\

\notag\\

=&\ \delta_0[c_0 
    \underbrace{
        (1 - \alpha_0)
    }_{\rlap{\text{$= T$ until the first sample point.}}} 
    - \overbrace{
        c_1\alpha_1(1 - \alpha_0) - ... - c_n\alpha_n(1 - \alpha_0) ... (1 - \alpha_{n - 1})
    }^{
        \substack{
            \text{The accumulated color until the first sample point:} \\
            \text{$=c - c_0\alpha_0 = \bar{c}$m}
        }
    }] \notag\\

\notag\\

=&\ \underbrace{\delta_0[c_0T - (c - \bar{c})]}_{\text{Holds for (r, g, b).}}\notag\\

\end{align}
$$
```