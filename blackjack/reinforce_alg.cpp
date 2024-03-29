#include "blackjack.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <cassert>
#include <iomanip>

void reinforcement_algorithm() {
  constexpr int Iteration = 10000000;
  constexpr double Gamma = 1;
  const double alpha = std::pow(2, -6);
  std::mt19937 mt(786);
  std::unordered_map<State, std::vector<double>> theta;

  for(int p_sum = 0; p_sum <= 21; p_sum++) {
    for(int p_ace = 0; p_ace <= 1; p_ace++) {
      if(p_sum == 21 && p_ace == 1) continue;
      for(int d = 1; d <= 10; d++) {
        Hand player = { p_sum, !!p_ace };
        Hand dealer = { d == 1 ? 0 : d, d == 1 };
        if(player.ace_update()) continue;
        int score = player.score();
        State state;
        state.player = player;
        state.dealer = dealer;
        theta[state] = std::vector<double>(2, 0);

        //std::cerr << player.sum << " " << player.usable_ace << " " << dealer.sum << " " << dealer.usable_ace << std::endl;
      }
    }
  }

  auto action_select = [&](const State& state) {
    double p = std::exp(theta[state][0]) / (std::exp(theta[state][0]) + std::exp(theta[state][1]));
    return std::bernoulli_distribution(p)(mt) ? 0 : 1;
  };
  for(int q = 0; q < Iteration; q++) {
    //std::cerr << "----" << std::endl;
    State state;
    state.init(mt);
    std::vector<std::tuple<State, int, double>> hist;
    while(true) {
      //std::cerr << state.player.sum << " " << state.player.usable_ace << " " << state.player.score() << " " << state.dealer.sum << " " << state.dealer.usable_ace << std::endl;
      int action = action_select(state);
      //std::cerr << action << std::endl;
      State before = state;
      auto [finished, R] = state.action(mt, action);
      //std::cerr << action << " " << finished << " " << R << std::endl;
      hist.push_back({ before, action, R });
      if(finished) break;
    }

    std::reverse(hist.begin(), hist.end());
    double G = std::get<2>(hist.front());
    for(auto& [S, A, R]: hist) {
      auto& state = S;
      //std::cerr << state.player.sum << " " << state.player.usable_ace << " " << state.player.score() << " " << state.dealer.sum << " " << state.dealer.usable_ace << " " << A << " " << R << std::endl;
      double pi_sum = std::exp(theta[S][0]) + std::exp(theta[S][1]);
      theta[S][A] += alpha * G * (1 - std::exp(theta[S][A]) / pi_sum);
      theta[S][1 - A] += alpha * G * (0 - std::exp(theta[S][1 - A]) / pi_sum);
    }

    if(q % 50000 == 0) {
      std::cerr << q << " " << Iteration << std::endl;
      std::map<std::tuple<int, int, int>, int> pi;
      for(auto& [s, v]: theta) {
        int a = std::max_element(v.begin(), v.end()) - v.begin();
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        pi[{ score, dealer, ace }] = a;
      }

      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << pi[{ i, d, ace }];
          }
          std::cerr << std::endl;
        }
      }

      /*

      std::map<std::tuple<int, int, int, int>, double> q;
      for(auto& [s, a]: theta) {
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        q[{ score, dealer, ace, 0 }] = a[0];
        q[{ score, dealer, ace, 1 }] = a[1];
      }


      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{ i, d, ace, 0 }] << "/";
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{i, d, ace, 1}] << " ";
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          }
          std::cerr << std::endl;
        }
      }
      */
    }
  }

}

int main() {
  //on_policy_eps_soft(0.1);
  reinforcement_algorithm();
}

/*
ace = 0
     A23456789T
11 : 1111111111
12 : 1111111111
13 : 1111111111
14 : 1111111111
15 : 1110001111
16 : 1000001111
17 : 1000000100
18 : 0000000000
19 : 0000000000
20 : 0000000000
21 : 0000000000
ace = 1
     A23456789T
11 : 1111111111
12 : 1111111111
13 : 1111111111
14 : 1111111111
15 : 1111111111
16 : 1111111111
17 : 1111111111
18 : 1000000011
19 : 0000000000
20 : 0000000000
21 : 0000000000
*/
