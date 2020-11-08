### 06.10.2020

* Discriminator doesn't learn with too few layers. With two layers it was not able to distinguish between fixed real and fixed fake state. It is able to do so with 5 layers.


### 08.10.2020

* The training regime should pick R or G at random, not first batch of R, then batch of G etc.
* How purity (or lack of it) of real input influence training? (Remember that $tr[p] = sum(\lamda) = 1$ and $tr[p^2] = sum(\lambda^2) \le 1$). How eigenvalues of real and generated state influence training.
