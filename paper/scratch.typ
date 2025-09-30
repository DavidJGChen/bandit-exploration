#import "@local/david:1.0.1": *

#show: david.with(
  class: "Summer 2025 Indepedent Study",
  title: [Bandit Alignment Scratch],
  student: "David J Chen",
  date: datetime.today(),
  subproblems: "1.1.a.i",
  collaborators: (),
  bibliography: bibliography("refs.bib"),
)

Let $epsilon.alt = abs(cal(A))^(1 slash 3) / T^(1 slash 3)$

$
  frak(R)(pi, T) &<= sqrt((abs(cal(A)) T (1 - epsilon.alt)^2) / epsilon.alt + 32 epsilon.alt^2 T^2) dot sqrt(abs(cal(A)) ln 4 T) \
  &<= sqrt((abs(cal(A)) T) / epsilon.alt + 32 epsilon.alt^2 T^2) dot sqrt(abs(cal(A)) ln 4 T) \
  &= sqrt((abs(cal(A)) T) / (cal(abs(A))^(1 slash 3) / T^(1 slash 3)) + 32 (cal(abs(A))^(1 slash 3) / T^(1 slash 3))^2 T^2) dot sqrt(abs(cal(A)) ln 4 T) \
  &= sqrt(abs(cal(A))^(2 slash 3) T^(4 slash 3) + 32 abs(cal(A))^(2 slash 3) T^(4 slash 3)) dot sqrt(abs(cal(A)) ln 4 T) \
  &<= sqrt(33 abs(cal(A))^(2 slash 3) T^(4 slash 3)) dot sqrt(abs(cal(A)) ln 4 T) \
  &<= sqrt(33 ln (4 T)) abs(cal(A))^(5 slash 6) T^(2 slash 3)\
$

Compare this to the result in the paper:

$
  frak(R)(pi, T) <= sqrt(33 ln (4 T)) abs(cal(A))^(3 slash 4) T^(3 slash 4)
$

Improved scaling with respect to the time horizon $T$, and worsened with respect to $cal(A)$, the size of the action set. This is an improvement as long as $epsilon.alt < 1 slash 2$.

// $
//   &(abs(cal(A))T) / (abs(cal(A))^(1 slash 3) / T^(1 slash 3)) + (abs(cal(A))^(1 slash 3) / T^(1 slash 3))^2 T^2 \
//   &= abs(cal(A))^(2 slash 3) T^(4 slash 3) + abs(cal(A))^(2 slash 3) T^(4 slash 3)
// $

// $
//   &(abs(cal(A))T) / (abs(cal(A))^(1 slash 2) / T^(1 slash 2)) + (abs(cal(A))^(1 slash 2) / T^(1 slash 2))^2 T^2 \
//   &= abs(cal(A))^(1 slash 2) T^(3 slash 2) + abs(cal(A))T
// $