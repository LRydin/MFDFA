---
title: 'MFDFA: Multifractal Detrended Fluctuation Analysis in Python'
tags:
  - Fluctuation Analysis
  - Multifractality
  - Self similarity
  - Nonstationary timeseries
authors:
 - name: Leonardo Rydin Gorjão
   orcid: 0000-0001-5513-0580
   affiliation: "1, 2"
affiliations:
 - name: Forschungszentrum Jülich, Institute for Energy and Climate Research - Systems Analysis and Technology Evaluation (IEK-STE), 52428 Jülich, Germany
   index: 1
 - name: Institute for Theoretical Physics, University of Cologne, 50937 Köln, Germany
   index: 2
date: 17th of December, 2019
bibliography: bib.bib
---

# Summary

A common tool to unveil the nature of the scaling and fractionality of a process, natural or computer-generated, is the Detrended Fluctuation Analysis (`DFA`), initally developed by Peng *et al.* and later extended to study multifractal processes by Kandelhardt *et al.*, giving rise to Multifractal Detrended Fluctuation Analysis (`MFDFA`) [@Peng1994;@Kantelhardt2002].
It addresses the question of the presence of long-range correlations and can be employ to study discrete processes [@Hurst1951], like auto-regressive models, as well as time-continuous stochastic processes.
An extensive study of DFA and the interplay between trends in data and correlated noise can be found in @Hu2001.

In order to determine the self-similarity of a stochastic process, one can study the relation between the variance of the process and time or space.
Auto-regressive and stochastic processes diffuse with different rates, and uncovering the rates of diffusion is of importance in natural processes with power-law correlations, like temperature variability [@Meyer2019], earthquake frequency [@Shadkhoo2009], and heartbeat dynamics [@Ivanov1999].
Fluctuation Analysis provides a method to uncover these correlations, but fails in the presence of trends in the data, which is, for example, particularly present in weather and climate data.
Detrending the data via polynominal fittings allows one to uncover solely the relation between the inherent fluctuations and the time scaling of a process.
Moreover, several processes might be driven by more than one time scale.
They might be of a mono- or multi-fractal nature.
By studying a continuum of power variations of the Detrended Fluctuation Analysis one extends into Multifractal Detrended Fluctuation Analysis, which permits the study of the fractality of the data by comparing power variations of Detrended Fluctuation Analysis.

There are currently no flexible and available implementations of Multifractal Detrended Fluctuation Analysis in `python`, but there are several `Matlab` versions available.
There is a particularly thorought introductory guide to Multifractal Detrended Fluctuation Analysis, and subsequently a source-code by Espen Ihlen from 2012 [@Ihlen2012], which is flexible but slow.
With this implementation, efficiency was sought, by making the most out of Python, reshaping the code to allow for multithreading, especially relying on `numpy`'s `polynomial`, which scales easily with modern computers having more CPU cores.

# Theory
Multifractal Detrended Fluctuation Analysis studies the fluctuation of a given process by considering increasing segments of the timeseries.
Take a timeseries $X(t)$ with $N$ elements $X_i$, $i=1,2, \dots, N$.
Obtained the *detrended* profile of the process by defining
$$
  Y_i = \sum_{k=1}^i \left ( X_k - \langle X \rangle \right),~\text{for}~i=1,2, \dots, N,\nonumber
$$
i.e., the cumulative sum of $X$ subtracting the mean $\langle X \rangle$ of the data.
Section the data into smaller non-overlapping segments of length $s$, obtaining therefore $N_s = \text{int}(N/s)$ segments.
Given the total length of the data isn't always a multiple of the segment's length $s$, discard the last points of the data.
Consider the same data, apply the same procedure, but discard now instead the first points of the data.
One has now $2N_s$ segments.

To each of this segments fit a polynomial $y_v$ of order $m$ and calculate the variance of the difference of the data to the polynomial fit
$$
  F^2(v,s) = \frac{1}{s} \sum_{i=1}^s [Y_{(v-1)s + i} - y_{v,i}]^2, ~\text{for}~v=1,2, \dots, N_s,\nonumber
$$
where $y_{v,i}$ is the polynomial fitting for the segment $i$ of length $v$.
One also has the freedom to choose the order of the polynomial fitting.
This gives rise to the denotes `DFA1`, `DFA2`, $\dots$, for the orders chosen.

Notice now $F^2(v,s)$ is a function of each variance of each $v$-segment of data and of the different $s$-length segments chosen.
One can now define, will all due generally, the $q$-th order fluctuation function by averaging each row of segments of size $s$
$$
  F_q^2(s) = \left\{\frac{1}{N_s} \sum_{v=1}^{N_s} [F^2(v,s)]^{q/2}\right\}^{1/q}\nonumber \tag{1}
$$
where Detrended Fluctuation Analysis with $q=2$ is a subset of Multifractal Detrended Fluctuation Analysis (where $q\in \mathbb{R}$).
The $q$-th order fluctuation function $F_q^2(s)$ is our object of interest.

The inherent scaling properties of the data, if the data displays power-law correlations, can now be studied in a log-log plot of $F_q^2(s)$ versus $s$, where the scaling of the data obeys a power-law with exponent $h(q)$ as
$$
  F^2(s) \sim s^{h(q)},\nonumber
$$
where $h(q)$ is the \textit{self-similarity} exponent, which may dependent on $q$, and relates directly to the Hurst index.
The self-similarity exponent $h(q)$ is calculated by finding the slope of this curve in the log-log plots, as seen henceforth in the figures.

If the data is monofractal, the `self-similarity` exponent $h(q)=h$ is independent of $q$.
On the other hand, if the data is multifractal, the dependence on $q$ can be understood by studying the multifractal scaling exponent $\tau(q)$
$$
  \tau(q) = qh(q) - 1.\nonumber \tag{2}
$$
For a clearer discussion of these properties, see [@Barabasi1991;@Kantelhardt2002]. If $\tau(q)$ is a linear function, the process is monofractal.

# Examples
To exemplify the usage of Multifractal Detrended Fluctuation Analysis, take two common examples of stochastic processes, a fractional Ornstein--Uhlenbeck process and general process that has a symmetric Lévy $\alpha$-stable distribution, with single parameter $\alpha$.

For an example of multifractal behaviour in real-world data of sun spots timeseries, alongside a detailed explanation of MFDFA, or an application to European temperature variability, see respectively @Movahed2006 and @Meyer2019.

To study the scaling effects in continuous stochastic processes, three exemplary fractional Ornstein--Uhlenbeck processes are taken, given by
$$
  dX_t = - \theta X_t dt + \sigma d B^H_t, \tag{3}
$$
with fixed $\theta=1.0$ and $\sigma=0.5$, but with a Hurst index of $H=0.3, 0.5, 0.7$.
Fractional Brownian motion self-similarity exponent is given by the Hurst index $H$, thus the three choices of fractional Ornstein--Uhlenbeck should result in a scaling of $h(q) = h = H+1$, where $+1$ is due to the integration of the process (i.e., and increase in regularity).
In Fig. 1 the MFDFA of the three processes can be seen.
The slopes of the curve in a log-log plot yield the self-similarity factor $h$ matching with the expected values.

![Fig. 1](fig1.pdf)
Fig. 1: Multifractal Detrended Fluctuation Analysis of three exemplary paths of fractional Ornstein--Uhlenbeck processes, given by Eq. (3),  with Hurst indices of $H=0.3, 0.5, 0.7$. Panel a) displays the log-log plot of the segment size $s$ versus the fluctuation function $F_q^2(s)$ [Eq. (1)], for $q=2$.
The inset shows $F_q^2(s)$ for the case of $H=0.3$ and the power variations $q=-10,-2,2,10$.
The lines are all parallel, indicating that the process is monofractal, as expected.
The dashed lines indicate the theoretical expected scalings.
The self-similarity scalings $h$ are obtained by extracting the slopes of the curves (in a log-log scale).
Panel b) shows the multifractal scaling exponent $\tau(q)$ [Eq. (2)] which exhibits a linear dependency, i.e. $h(q)=h$, indicating again the process is monofractal.
Numerically integrated with an integration step $\Delta t = 0.001$ over $100$ time units ($N=10^5$ data points).
The `MFDFA` algorithm ran in $2.67\!~$s$\pm 178\!~$ms, for $100$ segments $s$ and $40$ q-variation powers, with $2$ CPU cores.

Secondly, in the spirit of Kandelhardt \textit{et al}, take a process $X_t$ that has an symmetric $\alpha$-stable distribution, i.e., it has a power-law distribution function $P(x)\sim x^{-(\alpha-1)}$.
These processes exhibit heavy tails, infinite variances, and multifractal scaling.
In Fig. 2 three processes which are symmetric $\alpha$-stable distributed, with $\alpha = 1.75, 1.25,$ and $0.75$, are shown.
The multifractal behaviour can be identified directly in panel a), for $\alpha = 1.25$, given the curves of $F_q^2(s)$ are no longer parallel for different $q$-variations.

![Fig. 2](fig2.pdf)
Fig. 2: Multifractal Detrended Fluctuation Analysis of three exemplary symmetric Lévy $\alpha$-stable distributed processes, with $\alpha = 1.75, 1.25,$ and $0.75$, are shown.
In panel a) the $\alpha = 1.25$ Lévy distributed process is shown, and $F_q^2(s)$ for $q = -10, -5, -2, 2, 5, 10$ are displayed.
For $q>0$ the curves are not parallel, indicating the multifractal nature of the process.
Panel b) displays the \textit{self-similarity} exponent $h(q)$ , where a clear non-linear dependency on $q$ is observable.
The inset displays the multifractal scaling exponent $\tau(q)$ displaying two clear distinct behaviours for $q<0$ and $q>0$.
The dashed lines indicate the theoretical expected scalings.
The three process were drawn from Lévy $\alpha$-stable distributions, each with $N=10^6$ data points.
The `MFDFA` algorithm ran in $30.8\!~$s$\pm 940\!~$ms with $2$ CPU cores ($16.8\!~$s$\pm 91\!~$ms with $16$ CPU cores), for $100$ segments $s$ and $40$ q-variation powers

# The `MFDFA` library
The Multifractal Detrended Fluctuation Analysis library `MFDFA` presented is a standalone package based integrally on python's `numpy`, thus it can avail also of `numpy`'s masked arrays `ma`, which is particularly convenient when dealing with large datasets with missing data.

The `MFDFA` library offers a considerable speed-up in comparison with the available Matlab version.
The library is fully developed to work with multithreading, which shows an increase in performance while handling timeseries larger than $10^5$ data points, as seen in Fig. 3.

![Fig. 3](fig3.pdf)
Fig. 3: Performance of `MFDFA` in comparison with the distributed `Matlab` version [@Ihlen2012].
The `MFDFA` library and `Matlab` version were tested for a timeseries of a fractional Ornstein--Uhlenbeck [Eq. (3)] with increasing number of points $N$.
The `MFDFA` library is also tested with $2$ and $16$ cores.
Speed-ups are only noticeable above $10^5$ data points.
The Matlab code is truncated, since the computation times start exceeding $200$ seconds.
Both codes were tested for $20$ segments $s$ and $10$ q-variation powers.


# Acknowledgements
I'd like to thank Francisco Meirinhos for all the help with python, Fabian Harang, Marc Lagunas Merino, Anton Yurchenko-Tytarenko, Dennis Schroeder, Michele Giordano, Giulia di Nunno and Fred Espen Benth for all the help in understanding stochastic processes, and Dirk Witthaut for the unwavering support.
I gratefully acknowledge support by the Helmholtz Association, via the joint initiative \textit{Energy System 2050 - A Contribution of the Research Field Energy}, the grant No. VH-NG-1025, the scholarship funding from \textit{E.ON Stipendienfonds}, and the \textit{STORM - Stochastics for Time-Space Risk Models} project of the Research Council of Norway (RCN) No. 274410.

# References
