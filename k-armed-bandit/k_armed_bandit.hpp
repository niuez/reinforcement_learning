#pragma once
#include <vector>
#include <random>
#include <ostream>
#include <algorithm>

struct Bandit {
  double mean;
  double stddev;

  double change_mean = 0;
  double change_stddev = 0;

  template<class URGB>
  double reward(URGB& g) const {
    return std::normal_distribution<double>(mean, stddev)(g);
  }

  template<class URGB>
  void change(URGB& g) {
    mean += std::normal_distribution<double>(change_mean, change_stddev)(g);
  }

};

std::ostream& operator<<(std::ostream& os, const Bandit& bandit) {
  return os << "{ mean=" << bandit.mean << ", stddev=" << bandit.stddev << " }";
}

struct MultiBandit {
  std::vector<Bandit> bandits;

  template<class URGB, class BanditGenerator>
  MultiBandit(URGB& g, int n, BanditGenerator& bandit_gen, double bandit_stddev = 1, double change_mean = 0, double change_stddev = 0)
    : bandits(n) {
      for(int i = 0; i < n; i++) {
        bandits[i].mean = bandit_gen(g);
        bandits[i].stddev = bandit_stddev;
        bandits[i].change_mean = change_mean;
        bandits[i].change_stddev = change_stddev;
      }
  }

  int size() const {
    return bandits.size();
  }

  template<class URGB>
  double reward(URGB& g, int action) const {
    return bandits[action].reward(g);
  }

  template<class URGB>
  void change(URGB& g) {
    for(auto& bandit: bandits) {
      bandit.change(g);
    }
  }

  int best_action() const {
    return std::max_element(bandits.begin(), bandits.end(), [](const Bandit& a, const Bandit& b) { return a.mean < b.mean; }) - bandits.begin();
  }
};
std::ostream& operator<<(std::ostream& os, const MultiBandit& bandits) {
  for(int i = 0; i < bandits.bandits.size(); i++) {
    os << "#" << i << " " << bandits.bandits[i] << std::endl;
  }
  return os;
}
