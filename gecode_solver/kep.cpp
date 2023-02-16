// Unfinished code for solving the KEP with gecode solver

#include <gecode/int.hh>
#include <gecode/search.hh>
#include <gecode/driver.hh>

using namespace Gecode;

// TODO Problem pode virar KEP_Instance em um outro CPP
/// Problem instance
class KEP_Instance
{
private:
    const int _num_edges;
    const int _num_nodes;
    const int *_edge_weights;

public:
    /// Initialize problem instance
    KEP_Instance(const int num_edges, const int num_nodes, const int *edge_weights);
    /// Getters
    const int num_nodes(void) const;
    const int num_edges(void) const;
    const int *edge_weights(void) const;
    /// Return distance between node \a i and \a j
    int node_distance(int i, int j) const;
};
inline KEP_Instance::KEP_Instance(const int num_edges, const int num_nodes, const int *edge_weights) : _num_edges(num_edges), _num_nodes(num_nodes), _edge_weights(edge_weights) {}
inline const int KEP_Instance::num_nodes(void) const
{
    return _num_nodes;
}
inline const int KEP_Instance::num_edges(void) const
{
    return _num_edges;
}
inline const int *KEP_Instance::edge_weights(void) const
{
    return _edge_weights;
}
inline int KEP_Instance::node_distance(int i, int j) const
{
    return _edge_weights[i * _num_edges + j];
}

// class KidneyExchangeProblem : public Space
class KidneyExchangeProblem : public IntMinimizeScript
{
protected:
    /// KEP instance to be solved
    KEP_Instance instance;
    /// flow in array
    IntVarArray flow_in;
    /// flow out array
    IntVarArray flow_out;
    /// solution array
    IntVarArray y;
    /// Total score (sum of weighted donations in solution)
    IntVar score;

public:
    /// Actual model
    KidneyExchangeProblem(const SizeOptions &opt) : IntMinimizeScript(opt), y(*this, this->instance.num_edges(), 0, 1)
    {
        // get num_edges
        int num_edges = instance.num_edges();
        // edge weight matrix
        IntArgs w(num_edges * num_edges, instance.edge_weights());

        // TODO constraints
        // constraint 1a: sum(y[e], paratodo e em in(v)) = f_in[v], paratodo v em V
        int is_in_edge, node_flow_in;
        for (int node_index = 0; node_index < instance.num_nodes(); node_index++)
        {
            node_flow_in = 0;
            // for (int edge_index = 0; edge_index < instance.num_edges(); edge_index++)
            for (int src_node_index = 0; src_node_index < instance.num_nodes(); src_node_index++)
            {
                // current edge's dst is current node (node_index)
                if (instance.node_distance(src_node_index, node_index) != -1)
                {
                    node_flow_in++;
                }
            }
            rel(*this, flow_in[node_index], IRT_EQ, node_flow_in);
        }
        // constraint 1b: sum(y[e], paratodo e em out(v)) = f_out[v], paratodo v em V
        // constraint 2
        // f_out[v] <= f_in[v] <= 1, paratodo v em V
        // constraint 3
        // f_out[v] <= 1
        // constraint 4
        // TODO entender kk
        // sum(y[e], paratodo e em C) <= abs_size(c), paratodo C em C\Ck
        // constraint 5
        // o dominio dos y é {0,1}
    }

    /// Return solution score
    virtual IntVar score(void) const
    {
        return score;
    }
    // Print solution
    virtual void print(std::ostream &os) const
    {
        // TODO
        os << "Score: " << score << std::endl;
        os << "Number of edges in solution: TODO" << std::endl;
        os << "Edges in solution: TODO" << std::endl;
    }
};
