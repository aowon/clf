#include <chrono>
#include <cmath>
#include <iostream>
#include <ostream>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <conio.h>
#include <windows.h>

// Gnuplot class handles POSIX-Pipe-communikation with Gnuplot
#include "./gnuplot_i.hpp"

#define SLEEP_LGTH 2
#define NPOINTS 50

using namespace std;
using namespace chrono;

// Global
int num = 100;
vector<vector<double>> u1 = {{1, 1}};
vector<vector<double>> u2 = {{-1, -1}};
int epochs = 1000;
double lr = 0.05;

void sleep(int i) { Sleep(i * 1000); }
void wait_for_key();
int plot(vector<vector<double>> points, vector<vector<double>> w_init,
         vector<vector<double>> learned_w);

int plot(vector<vector<double>> points, vector<vector<double>> w_init,
         vector<vector<double>> learned_w) {
  Gnuplot::set_GNUPlotPath("../bin/");

  try {
    std::vector<double> x_1, x_2, y_1, y_2;
    for (int i = 0; i < 100; ++i) {
      x_1.push_back(points[i][0]);
      y_1.push_back(points[i][1]);
      x_2.push_back(points[199 - i][0]);
      y_2.push_back(points[199 - i][1]);
    }
    Gnuplot show;
    show.set_xrange(-4, 5).set_yrange(-4, 5).set_pointsize(0.5);
    show.set_grid();
    show.set_samples(600);
    show.set_style("points").plot_xy(x_1, y_1, "offset { 1, 1}");
    show.set_style("points").plot_xy(x_2, y_2, "offset {-1,-1}");
    show.set_style("lines");
    show.plot_slope(-w_init[1][0], w_init[0][0], "w init");
    show.plot_slope(-learned_w[1][0], learned_w[0][0], "learned w");
    show.set_surface().replot();
    wait_for_key();

  } catch (GnuplotException ge) {
    cout << ge.what() << endl;
  }
  return 0;
}

void wait_for_key() {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) ||                 \
    defined(__TOS_WIN__) // every keypress registered, also arrow keys
  cout << endl << "Press any key to continue..." << endl;

  FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
  _getch();
#elif defined(unix) || defined(__unix) || defined(__unix__) ||                 \
    defined(__APPLE__)
  cout << endl << "Press ENTER to continue..." << endl;

  std::cin.clear();
  std::cin.ignore(std::cin.rdbuf()->in_avail());
  std::cin.get();
#endif
  return;
}

vector<int> shape(vector<vector<double>> vec) {
  return {static_cast<int>(vec.size()), static_cast<int>(vec[0].size())};
}

vector<vector<double>> vector_add(vector<vector<double>> v1,
                                  vector<vector<double>> v2) {
  for (int i = 0; i < v1.size(); ++i) {
    for (int j = 0; j < 2; ++j) {
      v1[i][j] += v2[0][j];
    }
  }
  return v1;
}

vector<vector<double>> r_(vector<vector<double>> v1,
                          vector<vector<double>> v2) {
  for (int i = 0; i < v2.size(); ++i) {
    v1.push_back(v2[i]);
  }
  return v1;
}

vector<vector<double>> dot(vector<vector<double>> v1,
                           vector<vector<double>> v2) {
  vector<vector<double>> vec(v1.size(), vector<double>(v2[0].size()));
  for (int i = 0; i < v1.size(); ++i) {
    for (int j = 0; j < v2[0].size(); ++j) {
      vec[i][j] = 0;
      for (int k = 0; k < v1[0].size(); ++k) {
        vec[i][j] += (v1[i][k] * v2[k][j]);
      }
    }
  }
  return vec;
}

vector<vector<double>> randn(int row, int column) {
  static default_random_engine e(200);
  static normal_distribution<double> n(0, 1);
  vector<vector<double>> vec(row, vector<double>(column));
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < column; ++j) {
      vec[i][j] = n(e);
    }
  }
  return vec;
}

int randint(int low, int high) {
  if (low == 0) {
    return rand() % high;
  } else {
    return rand() % (high - low) + low;
  }
}

vector<double> asarray(vector<double> v1, vector<double> v2, int num) {
  vector<double> part2(num, v2[0]);
  vector<double> part1(num, v1[0]);
  part1.insert(part1.end(), part2.begin(), part2.end());
  return part1;
}

vector<vector<double>> update_weight(vector<vector<double>> w,
                                     vector<vector<double>> x, double y_predict,
                                     double y_k) {
  for (int i = 0; i < w.size(); ++i) {
    cout << w[i][0] << endl;
    w[i][0] += x[0][i] * lr * y_predict * (1 - y_predict) * (y_k - y_predict);
  }
  return w;
}

double eval(vector<vector<double>> X, vector<double> y,
            vector<vector<double>> w) {
  vector<vector<double>> y_predict = dot(X, w);
  double loss = 0.0;
  for (int i = 0; i < y.size(); ++i) {
    if ((y_predict[i][0] > 0.5 && y[i] > 0.5) ||
        (y_predict[i][0] <= 0.5 && y[i] <= 0.5))
      loss += 0;
    else
      loss += 1;
  }
  return loss;
}

double predict(vector<vector<double>> x, vector<vector<double>> w) {
  // assert((x.size()==1 && x[0].size()==1));
  double z = dot(x, w)[0][0];
  return 1.0 / (1 + exp(-z));
}

vector<vector<double>> train(vector<vector<double>> X, vector<double> y,
                             vector<vector<double>> w, double lr) {
  vector<int> n_d = shape(X);
  for (int i = 0; i < epochs; ++i) {
    int k = randint(0, n_d[0]);
    vector<vector<double>> x;
    x.push_back(X[k]);
    double y_predict = predict(x, w);
    double y_k = y[k];
    for (int i = 0; i < w.size(); ++i) {
      w[i][0] += x[0][i] * lr * y_predict * (1 - y_predict) * (y_k - y_predict);
    }
    // if(i%40 == 0)
    //   cout << "loss: " << eval(X, y, w) << endl;
  }
  return w;
}

int main() {
  auto start = system_clock::now();
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static default_random_engine e(seed);
  static normal_distribution<double> n(0, 1);
  vector<vector<double>> w_init = randn(2, 1);
  vector<vector<double>> X =
      r_(vector_add(randn(num, 2), u1), vector_add(randn(num, 2), u2));
  vector<double> y = asarray({1}, {0}, num);
  vector<vector<double>> learned_w = train(X, y, w_init, lr);

  auto finish = system_clock::now();
  auto duration =
      std::chrono::duration_cast<chrono::microseconds>(finish - start);
  cout << "Cost Time: " << duration.count() << " microseconds" << endl;
  cout << "loss: " << eval(X, y, learned_w) << endl;
  plot(X, w_init, learned_w);
}