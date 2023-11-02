#include <iostream>
#include <list>
#include <vector>
#include <queue>

template<class T>
struct graph {
public:
  graph(int v_num):v(v_num) {
    edge = std::vector<std::list<T>>(v_num);
    indegree = std::vector<int>(v_num, 0);
  }

  void addEdge(int from, int to) {
    edge[in].push_back(to);
    indegree[to]++;
  }

  int v;
  vector<int> indegree;
  std::vector<list<T>> edge;

}

template<class T>
void topological_sorting(graph<T> g) {
  std::queue<int> q;

  for (int i = 0; i < g.indegree.size(); ++i ) {
    if (g.indegree[i] == 0) {
      q.push[i];
    }
  }

  while(!q.empty()) {
    int v = q.top()
    std::cout << v << std::endl;
    q.pop();
    for (auto i = g.edge[v].begin(); i != g.edge[v].end(), ++i) {
      g.indegree[*i]--;
      if (g.indegree[*i] == 0) q.push(*i);
    }
  }
  return;
}

int main() {
}
