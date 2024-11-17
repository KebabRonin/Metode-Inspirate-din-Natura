## Lab 1
arhivx - vezi ce se publica acum
daca nu am facut genetici: 1' + 2 + 2'

codecamp product placement, mergi la vara ca e lume importanta

latex tikz pt diagrame

determinist probabilist euristic

30 rulari pt un data point

## Lab2

C/C++/Rust


De scris in english raportul

AG: prob de mutatie = 1 / nr gene
asteapta pana la selectie ca sa reduci pop
prob xover 50-80


cum init random?
* Moire pattern
* seed aux rng with time + thread id + argv, discard first 31337, then use that rng to init my actual rng
* what is mt19937_64 generator?
* use all bits from number (generate number take each n bits later, regenerate and discard unused bits if you run out)


xover:
1. rand sort candidates
2. 50% take last fara sot 50% discard

## Lab 4

HC 30 dim 100 iter 15s
AG 30 dim 100 popsize 100 0.5s / iteratie?
am belito rau (sau nu)

Raport:
media
stdev
min
media timp rulare
mediana dist interquartila ca alternativa

order by alg, dims, functions ca sa se uite usor

dont use I
intro = explicatie a raportului, ce am facut, direct la obiect, sa contina si ce era in abstract > 7 randuri
mai lung ca scade puncte

sectiuni cu subsectiuni, cu descriere scurta fiecare sectiune

Concluzie
mai lung ca scade puncte

Bibliografie - format arhivix sau ceva de bibtex

ctso? pt swarm optimization dampening si zgomot ca sa fie decent
read lit, experimente, 

sample size 5, 10??????????????????


H1' diagrama + explicatie pentru bazine de atractie

gradul unei conferinte:
1. search core conferences
search gecco in latest -> rank A
daca nu apare, sau e national/sub B (exclusiv), no bueno



Alte modificari:
eps greedy mutatie dinamica 
* de la 2/n la 1.8/n
* mutatie in functie de pozitia in cromozom (pt numere de ex, mutatie mai mare pe bitii semnificativi la inceput, la final pe cei nesemnificativi, pt explorare cu pas mai mare la inceput)
* hipermutatie - o generatie are 5-30% mutatie asa la bulan (mutatia maxim e 50%, ca dupa faci doar flip pe ce e < 50)
    * cand pop stagneaza
* mutatie adaptiva:
stdev fitness, daca creste stdev scade mutatia, la fel si invers

crossover - sortat dupa random, sau fitness

xover pick elitism, dupa xover intre 

plots ca sa gasim parametri sa ne putem da cu parerea

mai multe populatii (insular), mut xover in interiorul unei pop, + operator de migratie


operatori PSO, cauta dupa keywords no shit sherlock

intreaba de elitism + xover random
5% elitism ca trebe val mica, xover mare, xover intre toti, nu doar primii
xover rate??? - la noi e cu proportie cum am zis mai sus

se poate da merge la H1' si H1


## Lab sapt 7

PSO - nr reale

100 ok, 50 bun, 20 sa zicem

velocities, nr iter, dim swarm: in this order

ce i-a dat lui bine: 1, 2.5, 2.5


sapt asta fara curs, recup sapt 8 pe linkul de seminar