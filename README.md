# a breath of fresh AIr 🌪️

## Mistä kyse?

- Neuroverkkoja n stuff.
- Your classic pytorch, tensorflow knockoff (sucky at best).
- Tää on ihan vaa demo.
- Eräänlainen harjoits/kokeilu neuroverkkojen parissa.

## Vähän matematiikaa

### Aluksi perusjuttuja (määritelmiä)

Olkoon $l$ neuroverkossa oleva kerros, $N_l$ tason $l$ neuronien lukumäärä, $L$ ulostulokerros. 

Olkoon $l$ kerroksen painokertoimet 

$$W^l=\left(\begin{matrix}
w_{11}^l&\cdots&w_{1N_l}^l\\
\vdots&\ddots&\vdots\\
w_{N_{l-1}1}^l&\cdots&w^lN_{l-1}N_l
\end{matrix}\right),$$

jossa $w_{ij}^{l}$ viittaa $l-1$ kerroksen $i$:nnen ja $l$ kerroksen $j$:nnen neuronin väliseen painokertoimeen.

Vastaavasti vakiotermit

$$B^l=\left(\begin{matrix}
b_1^l&\cdots&b_{N_l}^l
\end{matrix}\right)$$

Lisäksi edellisen kerroksen $l-1$ aktivaatio

$$A^{l-1}=\left(\begin{matrix}
a_{11}^{l-1}&\cdots&a_{1N_{l-1}}^{l-1}\\
\vdots&\ddots&\vdots\\
a_{m1}^{l-1}&\cdots&a_{mN_{l-1}}^{l-1}
\end{matrix}\right),$$

jossa $a^{l-1}_{ij}$ viittaa $l-1$ kerroksen $j$:nnen neuronin aktivaatioon $i$:nnen opetussarjan (batch/mini-batch) opetusesimerkissä.

> _Huom_. opetussarja koostuu $m$ opetusesimerkistä.

Nyt saadaan kerroksen $l$ painoitettu syöte (weighted input)

$$Z^l=A^{l-1}W^l+B^l=\left(\begin{matrix}
z_{11}^l&\cdots&z_{1N_l}^l\\
\vdots&\ddots&\vdots\\
z_{m1}^l&\cdots&z_{mN_l}^l
\end{matrix}\right)$$

>_Huom_. $B^l$ on $1 \times N_l$ kun taas $A^{l-1}W^l$ on $m \times N_l$. 
<br>Kuitenkin esim. NumPy:ssa summaus tapahtuu siten, että vektori $B^l$ lisätään jokaiseen $A^{l-1}W^l$ riviin. 

<br>

Kerroksen $l$ aktivaatio

$$A^l=\varphi_{l} (Z^l)=\left(\begin{matrix}
\varphi (z_{11}^l)&\cdots&\varphi (z_{1N_l}^l)\\
\vdots&\ddots&\vdots\\
\varphi (z_{m1}^l)&\cdots&\varphi (z_{mN_l}^l)
\end{matrix}\right),$$

jossa $\varphi_{l}$ on kerroksen $l$ aktivointifunktio.

<br>

Usein muissa teksteissä $z$-arvoja ja muita merkataan vain yhdellä indeksillä. Kuitenkin tässä tapauksessa kaikki pohjautuvat opetussarjoihin, jolloin yhden indeksin sijaan käytetään kahta. Ensimmäinen tyypillisesti kuvaa opetusesimerkin indeksiä ja toinen varsinaisen arvon indeksiä.

>Esim. $z_{ij}^{l}$ kuvastaa opetussarjan $i$:nnestä opetusesimerkistä tason $l$ neuronin $j$ painottetua arvoa.

<br>

#### Virhefunktiosta hieman

Olkoon $C$ virhefunktio, jonka tehtävä on arvioida neuroverkon toiminnan laatua.
<br>
Virhefunktio voi esimerkiksi olla:

$$C=\frac{1}{2m}\sum_{k=1}^{m}||A_k^L-Y_k||^2,$$

jossa $Y_k$ on $k$:nnen opetusesimerkin odotettu arvo ja $||A_k^L-Y_k||^2=\sum_{t=1}^{N_L} (a_{kt}^L-y_{kt})^2$ (ns. erotuksen euklidisen normin neliö)

<br>

#### Vastavirtamenetelmän neljä tärkeää derivaattaa 

Tarkastellaan kuinka $L$ kerroksen painoitettu sisääntulo (weighted input) vaikuttaa virhefunktioon.
<br>
$$\frac{\partial C}{\partial z^L_{ij}}=\frac{\partial C}{\partial a^L_{ij}} \frac{\partial a^L_{ij}}{\partial z^L_{ij}}=\frac{\partial C}{\partial a^L_{ij}} \varphi '(z^L_{ij})$$
<br>
Tätä arvoa usein nimitetään $L$ kerroksen $j$:nnen neuronin aiheuttamaksi virheeksi, jota usein merkataan $\delta_j^L$.
<br>
Merkataan kuitenkin arvoa mieluummin $\delta_{ij}^L$ , jossa $i$ viittaa opetusesimerkkiin.
<br>
Huomaa $\frac{\partial C}{\partial a_{ij}^L}$ tarkoittaa itse virhefunktion derivaattaa. 
<br>
Edellä mainitun virhefunktion tapauksessa se olisi $\frac{\partial C}{\partial a_{ij}^L}=\frac{1}{m}(a_{ij}^L-y_{ij})$

Sama voidaan esittää matriiseilla.

$$\delta^L=\left(\begin{matrix}
\nabla_{a^L_{1}}C\\
\nabla_{a^L_{2}}C\\
\vdots\\
\nabla_{a^L_{m}}C
\end{matrix}\right)\oplus\varphi'\left(Z^L\right),\tag{1}$$

jossa $\nabla_{a^L_i} C$ voidaan ajatella virhefunktion gradienttivektoriksi, jossa elementit ovat virhefunktion osittaisderivaattoja $\frac{\partial C}{\partial a_{ij}^L}$.

Operaatio $\oplus$ tarkoittaa ns. Hadamardin tuloa. Käytänössä siis matriisien _elementwise_ kertominen.

Tarkastellaan seuraavaksi kuinka $l$ kerroksen $z_{j}^{l}$ vaikuttaa virhefunktioon.

$$\begin{align}
\delta_{ij}^l&=\frac{\partial C}{\partial z_{ij}^{l}}=\sum_{k=1}^{N_{l+1}}\frac{\partial C}{\partial z_{ik}^{l+1}} \frac{\partial z_{ik}^{l+1}}{\partial z_{ij}^l}=\sum_{k=1}^{N_{l+1}}\delta_{ik}^{l+1} \frac{\partial z_{ik}^{l+1}}{\partial z_{ij}^l}\\
&=\varphi'(z_{ij}^l)\sum_{k=1}^{N_{l+1}}\delta_{ik}^{l+1}w_{jk}^{l+1}
\end{align}$$

Sama voidaan esittää matriiseilla.

$$\delta^{l}=\varphi'(Z^l) \oplus \delta^{l+1}(W^{l+1})^T\tag{2}$$

Lopulta saadaan virhefunktion osittaisderivaatat painokertoimien $w_{ij}^{l}$ ja vakiotermien $b_{j}^l$ suhteen.

$$\begin{align}
\frac{\partial C}{\partial w_{ij}^{l}}&=\frac{1}{m}\sum_{k=1}^{m}\delta_{kj}^{l}a_{ki}^{l-1}\\
\frac{\partial C}{\partial b_{j}^{l}}&=\frac{1}{m}\sum_{k=1}^{m}\delta_{kj}^l
\end{align}$$

Sama voidaan esittää matriiseilla.

$$\begin{align}
\frac{\partial C}{\partial W^l}&=\frac{1}{m}(A^{l-1})^T\delta^l\tag{3}\\
\frac{\partial C}{\partial B^l}&=\frac{1}{m}\delta^l\tag{4}\\
\end{align}$$

Näin saadut matriisit ovat siis opetusesimerkkien painokertoimien ja vakiotermien derivaattojen keskiarvot kyseiselle kerrokselle.

Kootaan vielä nämä neljä lauseketta.

$$\begin{align}\delta^L&=\left(\begin{matrix}
\nabla_{a^L_{1}}C\\
\nabla_{a^L_{2}}C\\
\vdots\\
\nabla_{a^L_{m}}C
\end{matrix}\right)\oplus\varphi'\left(Z^L\right)\tag{1}\\
\quad\\
\delta^{l}&=\varphi'(Z^l) \oplus \delta^{l+1}(W^{l+1})^T\tag{2}\\
\quad\\
\frac{\partial C}{\partial W^l}&=\frac{1}{m}(A^{l-1})^T\delta^l\tag{3}\\
\quad\\
\frac{\partial C}{\partial B^l}&=\frac{1}{m}\delta^l\tag{4}\\
\end{align}$$

### Vastavirta-algoritmi

Kaavoista nähdään, että $\delta^l$-termejä hyödynnetään aina edellisessä $l-1$ kerroksen derivaatoissa.
<br>
Siispä neuroverkon iterointi käänteisessä järjestyksessä on luontevampaa ja tehokkaampaa.
<br>
Vastavirta-algoritmissa jokaisen kerroksen kohdalla täytyy:

1. Laskea kerrosta vastaava $\delta^l$-arvo hyödyntäen aikaisemman kerroksen $\delta^{l-1}$-arvoa.
2. Laskea kerrosta vastaavat $\frac{\partial C}{\partial W^l}$ ja $\frac{\partial C}{\partial B^l}$
3. Päivittää painokertoimia ja vakiotermejä, jolloin $W^l=W_{\text{ennen}}^l-\eta\frac{\partial C}{\partial W^L}$ ja $B^l=B_{\text{ennen}}^l-\eta\frac{\partial C}{\partial B^l}$

### Lähteitä:

1. [Johdatus tekoälyn taustalla olevaan matematiikkaan, Heli Tuominen](https://tim.jyu.fi/view/143092#DKUvbnUuGytQ)

2. [How the backpropagation algorithm works, Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html)


