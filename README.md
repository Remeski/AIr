# a breath of fresh AIr 🌪️

Neuroverkkoja n stuff.

Your classic pytorch, tensorflow knockoff.

### Vähän matematiikaa

##### Aluksi perusjutut

Olkoon $N_l$ tason $l$ neuronien lukumäärä.

Olkoon $L$ ulostulokerros.

Olkoon painokertoimet $l$ kerrokselle

$$W^l=\left(\begin{matrix}
w_{11}^l&\cdots&w_{1N_l}^l\\
\vdots&\ddots&\vdots\\
w_{N_{l-1}1}^l&\cdots&w^lN_{l-1}N_l
\end{matrix}\right),$$

jossa $w_{ij}^{l}$ viittaa $l-1$ tason $i$:nnen ja $l$ tason $j$:nnen neuronin väliseen painokertoimeen.

Vastaavasti vakiotermit

$$B^l=\left(\begin{matrix}
b_1^l&\cdots&b_{N_l}^l
\end{matrix}\right)$$

Lisäksi edellisen kerroksen $l-1$ aktivaatio

$$A^{l-1}=\left(\begin{matrix}
a_{11}^{l-1}&\cdots&a_{1N_{l-1}}^{l-1}\\
\vdots&\ddots&\vdots\\
a_{m1}^{l-1}&\cdots&a^{l-1}_{mN_{l-1}}
\end{matrix}\right),$$

jossa $a^{l-1}_{ij}$ viittaa $l-1$ tason $j$:nnen neuronin aktivaatioon $i$:nnen opetussarjan (batch) opetusesimerkissä.

> _Huom_. opetussarja koostuu $m$ opetusesimerkistä.

Nyt saadaan tason $l$ painoitettu sisääntulo (weighted input)

$$Z^l=A^{l-1}W^l+B^l=\left(\begin{matrix}
z_{11}^l&\cdots&z_{1N_l}^l\\
\vdots&\ddots&\vdots\\
z_{m1}^l&\cdots&z_{mN_l}^l
\end{matrix}\right)$$

>_Huom_. $B^l$ on $1 \times N_l$ kun taas $A^{l-1}W^l$ on $m \times N_l$. 
<br>Kuitenkin esim. NumPy:ssa summaus tapahtuu siten, että vektori $B^l$ lisätään jokaiseen $A^{l-1}W^l$ riviin. 

Kerroksen aktivaatio

$$A^l=\varphi_{l} (Z^l),$$

jossa $\varphi_{l}$ on tason $l$ aktivointifunktio.

>_Huom_. Aktivointifunktio on _elementwise_, eli <br>
$\varphi_l (Z^l)=\left(\begin{matrix}\varphi (z^{l}_{11})&\cdots&\varphi (z^l_{1N_{l}})\\\vdots&\ddots&\vdots\\\varphi (z^{l}_{m1})&\dots&\varphi (z^l_{1N_{l}})\\\end{matrix}\right)$

<br>

Usein muissa teksteissä z-arvoja ja muita merkataan vain yhdellä indeksillä. Kuitenkin tässä tapauksessa kaikki pohjautuvat opetussarjoihin, jolloin yhden indeksin sijaan käytetään kahta. Ensimmäinen tyypillisesti kuvaa opetusesimerkin indeksiä ja toinen varsinaisen arvon indeksiä.


>Esim. $z_{ij}^{l}$ kuvastaa opetussarjan $i$:nnestä opetusesimerkistä tason $l$ neuronin $j$ painottetua arvoa.

<br>

##### Virhefunktiosta hieman

Olkoon $C$ virhefunktio, jonka tehtävä on määrittää neuroverkon toiminnan laatua.

Virhefunktio voi esimerkiksi olla:

$$C=\frac{1}{2}\sum_{k \in Y}(a^L_k-y_k)^2,$$

jossa $Y$ on joukko opetusesimerkkien vastuksia. 

##### Vastavirtamenetelmän neljä tärkeää derivaattaa 

Tarkastellaan kuinka $L$ kerroksen painoitettu sisääntulo (weighted input) vaikuttaa virhefunktioon.

$$\frac{\partial C}{\partial z^L_{ij}}=\frac{\partial C}{\partial a^L_{ij}} \frac{\partial a^L_{ij}}{\partial z^L_{ij}}=\frac{\partial C}{\partial a^L_{ij}} \varphi '(z^L_{ij})$$

Tätä arvoa usein nimitetään $L$ kerroksen $j$:nnen neuronin aiheuttamaksi virheeksi, jota usein merkataan $\delta_j^L$.

Merkataan kuitenkin arvoa mieluummin $\delta_{ij}^L$ , jossa $i$ viittaa opetusesimerkkiin.

Sama voidaan esittää matriiseilla.

$$\delta^L=\left(\begin{matrix}
\nabla_{a^L_{1}}C\\
\nabla_{a^L_{2}}C\\
\vdots\\
\nabla_{a^L_{m}}C
\end{matrix}\right)\oplus\varphi'\left(Z^L\right),\tag{1}$$

jossa $\nabla_{a^L_i} C$ voidaan ajatella virhefunktion gradienttivektoriksi, jossa elementit ovat virhefunktion osittaisderivaattoja $\frac{\partial C}{\partial a_{ij}^l}$.

Operaatio $\oplus$ tarkoittaa ns. Hadamardin tuloa. Käytänössä siis matriisien _elementwise_ kertominen.

Tarkastellaan seuraavaksi kuinka $l$ kerroksen $z_{j}^{l}$ vaikuttaa virhefunktioon.

$$\begin{align}\delta_{ij}^l&=\frac{\partial C}{\partial z_{ij}^{l}}=\sum_{k=1}^{N_{l+1}}\frac{\partial C}{\partial z_{ik}^{l+1}} \frac{\partial z_{ik}^{l+1}}{\partial z_{ij}^l}=\sum_{k=1}^{N_{l+1}}\delta_{ik}^{l+1} \frac{\partial z_{ik}^{l+1}}{\partial z_{ij}^l}\notag\\
&=\varphi'(z_{ij}^l)\sum_{k=1}^{N_{l+1}}\delta_{ik}^{l+1}w_{jk}^{l+1}\notag\end{align}$$

Sama voidaan esittää matriiseilla.

$$\delta^{l}=\varphi'(Z^l) \oplus \delta^{l+1}(W^{l+1})^T\tag{2}$$

Lopulta saadaan virhefunktion osittaisderivaatat painokertoimien $w_{ij}^{l}$ ja vakiotermien $b_{j}^l$ suhteen.

$$\begin{align}\frac{\partial C}{\partial w_{ij}^{l}}&=\frac{1}{m}\sum_{k=1}^{m}\delta^{l}_{kj}a_{ki}^{l-1}\notag\\
\frac{\partial C}{\partial b_{j}^{l}}&=\frac{1}{m}\sum_{k=1}^{m}\delta_{kj}^l\notag\end{align}$$

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
\end{matrix}\right)\oplus\varphi'\left(Z^L\right)\tag{1}\\\notag\\
\delta^{l}&=\varphi'(Z^l) \oplus \delta^{l+1}(W^{l+1})^T\tag{2}\\\notag\\
\frac{\partial C}{\partial W^l}&=\frac{1}{m}(A^{l-1})^T\delta^l\tag{3}\\\notag\\
\frac{\partial C}{\partial B^l}&=\frac{1}{m}\delta^l\tag{4}\\
\end{align}$$

#### Vastavirtamenetelmä

Kaavoista nähdään, että $\delta^l$-termejä hyödynnetään aina edellisessä $l-1$ kerroksen derivaatoissa.

Siispä neuroverkon iterointi käänteisessä järjestyksessä on luontevampaa ja tehokkaampaa.

Vastavirta-algoritmissa yhden kerroksen kohdalla täytyy:

1. Laskea kerrosta vastaava $\delta^l$-arvo hyödyntäen aikaisemman kerroksen $\delta^{l-1}$-arvoa.
2. Laskea kerrosta vastaavat $\frac{\partial C}{\partial W^l}$ ja $\frac{\partial C}{\partial B^l}$
