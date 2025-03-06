\documentclass{article}
\usepackage{amsmath}

\begin{document}

\section*{Sod Shock Tube Analytic Solution}
The Sod shock tube problem is defined with initial conditions:
\begin{align*}
\text{Left state (} x < 0 \text{):} &\quad \rho_L = 1.0, \quad p_L = 1.0, \quad v_L = 0.0 \\
\text{Right state (} x > 0 \text{):} &\quad \rho_R = 0.125, \quad p_R = 0.1, \quad v_R = 0.0
\end{align*}
where \(\gamma = 1.4\).

\subsection*{Regions}
The solution consists of five regions separated by:
1. Rarefaction fan left edge: \( x = -c_L t \), where \( c_L = \sqrt{\gamma p_L / \rho_L} \)
2. Contact discontinuity: \( x = v_L t \) (here, \( v_L = 0 \))
3. Shock front: \( x = V_s t \), where the shock speed is:
\[
V_s = c_L \sqrt{\frac{1 - \mu^2}{\gamma} \cdot \frac{p_R / p_L - 1}{1 - \mu^2 p_R / p_L}}, \quad \mu = \frac{\gamma - 1}{\gamma + 1}
\]

\subsection*{Solution}
\begin{itemize}
    \item \textbf{Left state (\( x < -c_L t \)):}
    \[
    \rho = \rho_L, \quad p = p_L, \quad v = v_L
    \]
    \item \textbf{Rarefaction fan (\( -c_L t \leq x \leq 0 \)):}
    \[
    v = \frac{1 + \mu}{1 - \mu} \left( \frac{x}{t} + c_L \right), \quad
    \rho = \rho_L \left( 1 - \mu \frac{v}{c_L} \right)^{\frac{2}{\gamma - 1}}, \quad
    p = p_L \left( \frac{\rho}{\rho_L} \right)^\gamma
    \]
    \item \textbf{Middle state (between contact and shock):}
    \[
    p_m = p_R \left[ 1 + \frac{2\gamma}{\gamma + 1} \left( \frac{V_s^2}{c_R^2} - 1 \right) \right],
    \quad \rho_m = \rho_R \frac{p_m / p_R + \mu}{1 + \mu p_m / p_R},
    \quad v_m = V_s \frac{\rho_R}{\rho_m}
    \]
    \item \textbf{Right state (\( x > V_s t \)):}
    \[
    \rho = \rho_R, \quad p = p_R, \quad v = v_R
    \]
\end{itemize}

\end{document}