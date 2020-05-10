# quantum-computing-resources

## Groups
* [QAR-Lab LMU](http://www.mobile.ifi.lmu.de/en/qar-lab/)
* [INnsbruck University](https://www.uibk.ac.at/th-physik/qic-group/research/)
* [PlanQK](http://planqk.de/en/) - Platform and ecosystem for Quantum-inspired Artificial Intelligence. The goal is to develop an open platform for Quantum-inspired AI to create and promote an ecosystem of AI & Quantum Computing (QC) specialists, developers of QC applications as well as users, service providers and consultants.
* [QUTech TU Delft] - https://qutech.nl/

## People
* Simon Benjamin - Oxford https://qtechtheory.org/
* Steven Flammia - UoS https://www.sydney.edu.au/science/about/our-people/academic-staff/steven-flammia.html#collapseprofileresearchinterest
* Stephen Bartlett - UoS https://www.sydney.edu.au/science/about/our-people/academic-staff/stephen-bartlett.html 
* Aram W. Harrow - MIT http://web.mit.edu/aram/www/


## Papers

### Overviews
* [The Holy Grail of Quantum Artificial Intelligence:Major Challenges in Accelerating the Machine Learning Pipeline](https://arxiv.org/pdf/2004.14035.pdf)
* [Opportunities and challenges for quantum-assisted machine learning in near-term quantum computers](https://arxiv.org/pdf/1708.09757.pdf)
* [Quantum Computing in the NISQ era and beyond](https://arxiv.org/pdf/1801.00862.pdf)

### Quantum error correction
* [A high threshold code for modular hardware with asymmetric noise](https://arxiv.org/pdf/1812.01505.pdf)
* [Fault-Tolerant Logical Gates in the IBM Quantum Experience](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.080504)
* [Multiqubit randomized benchmarking using few samples](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.032304)


### Variational Qauntum Algorithms
* [Variational-State Quantum Metrology](https://arxiv.org/pdf/1908.08904.pdf)
* [Variational quantum simulation of general processes](https://arxiv.org/pdf/1812.08778.pdf)
* [Theory of variational quantum simulation](https://arxiv.org/pdf/1812.08767.pdf)

### Quantum Compilers
* [Quantum compilation and circuit optimisation via energy dissipation](https://arxiv.org/pdf/1811.03147.pdf)


### Quantum Security
* [Simple proof of confidentiality for private quantum channels in noisy environments](https://arxiv.org/pdf/1711.08897.pdf)

### Handling Big Datasets
* [Small quantum computers and large classical data sets](https://arxiv.org/pdf/2004.00026.pdf)
## Software 

### [Qiskit](https://qiskit.org/)

Qiskit is an open-source quantum computing software development framework for leveraging today's quantum processors in research, education, and business.

### [PennyLane](https://pennylane.ai/)

A cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.d

## Notes

### Variational Quantum Algorithms(VQAs)

Variational quantum circuits by construction depend only on a linear or polynomial
number of parameters while the Hilbert-space dimension of the underlying quantum
state increases exponentially in the number of qubits. This advantageous scaling allows
one to tackle classically intractable problems. The general concept of variational
quantum algorithms is to prepare a parametrised quantum state using a quantum
processor and to vary its parameters externally until the optimum of a suitable cost
function is reached. This cost function can be tailored to the particular problem. For
example, one can search for the ground state of a molecule by setting the cost function to
be the expectation value of the corresponding molecular Hamiltonian. This technique is 
usually referred to as the variational quantum eigensolver (VQE) [1, 2, 3, 4]. Quantum
machine learning is another area where variational techniques may be valuable. One is
then interested in optimising a cost function that quantifies how similar the output of
the quantum circuit is to a fixed dataset [7, 8]. Moreover, it is also possible to recompile
a quantum circuit into another by optimising a metric on related quantum states [9, 10]. [Variational-State Quantum Metrology](https://arxiv.org/pdf/1908.08904.pdf)


Variational algorithms have been developed as a powerful classical tool for simulating manybody quantum systems. The core idea is based on
the intuition that physical states with low energy generally belong to an exponentially small manifold of the
whole Hilbert space


Variational simulation is a widely used technique in many-body physics 

### Optimization

Purely quantum methods differ from classical stochastic optimization in that
they are usually guaranteed to find the global optimum under ideal conditions. 
In real-world implementations, they, too, yield stochastic results.


## Possible topics
* Provide means to process (the essence of) large amounts of data on quantum computers.
* Provide standardized interfaces that allow for dynamic combination of QAI components and (by extension) for experts of different fields to collaborate on QAI algorithms.
* Focus on problems that are currently hard and intractable for the ML community, for example, generative models in unsupervised and semi-supervised (https://arxiv.org/pdf/1708.09757.pdf)
* Focus on datasets with potentially intrinsic quantum-like correlations, making quantum computers indispensable; these will provide the most compact and efficient model representation, with the potential of a significant quantum advantage even at the level of 50-100 qubit devices. (https://arxiv.org/pdf/1708.09757.pdf)
* Focus on hybrid algorithms where a quantum routine is executed in the intractable step of the classical ML algorithmic pipeline (https://arxiv.org/pdf/1708.09757.pdf)


## Questions
* Research in this field has been focusing on tasks such
as classification [14], regression [11, 15, 18], Gaussian
models [16], vector quantization [13], principal component analysis [17] and other strategies that are routinely
used by ML practitioners nowadays. We do not think
these approaches would be of practical use in near-term
quantum computers. The same reasons that make these
techniques so popular, e.g., their scalability and algorithmic efficiency in tackling huge datasets, make them less
appealing to become top candidates as killer applications
in QAML with devices in the range of 100-1000 qubits.
In other words, regardless of the claims about polynomial
and even exponential algorithmic speedup, reaching interesting industrial scale applications would require millions or even billions of qubits. Such an advantage is then
moot when dealing with real-world datasets and with the
quantum devices to become available in the next years in
the few thousands-of-qubits regime (source: https://arxiv.org/pdf/1708.09757.pdf)


## Other Links
* https://thequantumdaily.com/2019/11/18/the-worlds-top-12-quantum-computing-research-universities/
