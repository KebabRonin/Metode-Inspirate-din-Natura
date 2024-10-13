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

## Questions
1. Split [-10, 10] with precision = 1 in 200 or in 256 intervals?