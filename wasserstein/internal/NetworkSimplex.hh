//------------------------------------------------------------------------
// This file is part of Wasserstein, a C++ library with a Python wrapper
// that computes the Wasserstein/EMD distance. NetworkSimplex.hh is based
// on code from the LEMON graph library, which has been modified by
// Nicolas Boneel and subsequently Rémi Flamary. Their copyright notices
// appear below this one. If you use it for academic research, please cite
// or acknowledge the following works:
//
//   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
//       https://doi.org/10.1103/PhysRevLett.123.041801
//   - Komiske, Metodiev, Thaler (2020) arXiv:2004.04159
//       https://doi.org/10.1007/JHEP07%282020%29006
//   - Boneel, van de Panne, Paris, Heidrich (2011)
//       https://doi.org/10.1145/2070781.2024192
//   - LEMON graph library https://lemon.cs.elte.hu/trac/lemon
//
// Copyright (C) 2019-2021 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//------------------------------------------------------------------------

// LICENSE from POT: https://github.com/PythonOT/POT/blob/master/LICENSE
/*
MIT License

Copyright (c) 2016 Rémi Flamary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Copyright notice from network_simplex_simple.h
/* -*- mode: C++; indent-tabs-mode: nil; -*-
*
*
* This file has been adapted by Nicolas Bonneel (2013),
* from network_simplex.h from LEMON, a generic C++ optimization library,
* to implement a lightweight network simplex for mass transport, more
* memory efficient than the original file. A previous version of this file
* is used as part of the Displacement Interpolation project,
* Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
*
* Revisions:
* March 2015: added OpenMP parallelization
* March 2017: included Antoine Rolet's trick to make it more robust
* April 2018: IMPORTANT bug fix + uses 64bit integers (slightly slower but 
* less risks of overflows), updated to a newer version of the algo by LEMON,
* sparse flow by default + minor edits.
*
*
**** Original file Copyright Notice :
*
* Copyright (C) 2003-2010
* Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
* (Egervary Research Group on Combinatorial Optimization, EGRES).
*
* Permission to use, modify and distribute this software is granted
* provided that this copyright notice appears in all copies. For
* precise terms see the accompanying LICENSE file.
*
* This software is provided "AS IS" with no warranty of any kind,
* express or implied, and with no claim as to its suitability for any
* purpose.
*
*/

// Copyright notice from full_bipartite.h
/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 * This file has been adapted by Nicolas Bonneel (2013), 
 * from full_graph.h from LEMON, a generic C++ optimization library,
 * to implement a lightweight fully connected bipartite graph. A previous
 * version of this file is used as part of the Displacement Interpolation 
 * project, 
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
 * 
 *
 **** Original file Copyright Notice :
 * Copyright (C) 2003-2010
 * Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
 * (Egervary Research Group on Combinatorial Optimization, EGRES).
 *
 * Permission to use, modify and distribute this software is granted
 * provided that this copyright notice appears in all copies. For
 * precise terms see the accompanying LICENSE file.
 *
 * This software is provided "AS IS" with no warranty of any kind,
 * express or implied, and with no claim as to its suitability for any
 * purpose.
 *
 */

/*  _   _ ______ _________          ______  _____  _  __
 * | \ | |  ____|__   __\ \        / / __ \|  __ \| |/ /
 * |  \| | |__     | |   \ \  /\  / / |  | | |__) | ' /
 * | . ` |  __|    | |    \ \/  \/ /| |  | |  _  /|  <
 * | |\  | |____   | |     \  /\  / | |__| | | \ \| . \
 * |_| \_|______|  |_|      \/  \/   \____/|_|  \_\_|\_\
 *   _____ _____ __  __ _____  _      ________   __
 *  / ____|_   _|  \/  |  __ \| |    |  ____\ \ / /
 * | (___   | | | \  / | |__) | |    | |__   \ V /
 *  \___ \  | | | |\/| |  ___/| |    |  __|   > <
 *  ____) |_| |_| |  | | |    | |____| |____ / . \
 * |_____/|_____|_|  |_|_|    |______|______/_/ \_\
 */

#ifndef LEMON_NETWORK_SIMPLEX_HH
#define LEMON_NETWORK_SIMPLEX_HH

// C++ standard library
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

#include "EMDUtils.hh"

namespace lemon {

typedef EMDNAMESPACE::EMDStatus NetworkSimplexStatus;

using EMDNAMESPACE::free_vector;

//-----------------------------------------------------------------------------
// NetworkSimplex
//-----------------------------------------------------------------------------

// main parameters of block search pivot rule, consider changing these to check for speed
const double BLOCK_SIZE_FACTOR = 1.0;
const int MIN_BLOCK_SIZE = 10;
const int INVALID = -1;
const double INVALID_COST_VALUE = -1.0;

// templated NetworkSimplex class
// - Node: signed integer type that indexes particles
// - Arc: signed integer type that (roughly) can hold the product of two Nodes
// - Value: floating point type that is used for computations
// - Bool: boolean type (often not "bool" to avoid std::vector<bool> being slow)
template<typename N = int, typename A = long long, typename V = double, typename B = char>
class NetworkSimplex {
public:

  // datatype typedefs
  typedef N Node;
  typedef A Arc;
  typedef V Value;
  typedef B Bool;

  // rough type checking
  static_assert(std::is_integral<Node>::value && std::is_signed<Node>::value,
                "Node should be a signed integral type.");
  static_assert(std::is_integral<Arc>::value && std::is_signed<Arc>::value,
                "Arc should be a signed integral type.");
  static_assert(std::is_floating_point<Value>::value, "Value should be a floating point type.");
  static_assert(std::is_integral<Bool>::value, "Bool should be an integral type.");

  // vector typedefs
  typedef std::vector<Node> NodeVector;
  typedef std::vector<Arc> ArcVector;
  typedef std::vector<Value> ValueVector;
  typedef std::vector<Bool> BoolVector;
  typedef std::vector<char> StateVector;

  // constructor
  NetworkSimplex(unsigned n_iter_max, Value epsilon_large_factor, Value epsilon_small_factor) :
    MAX(std::numeric_limits<Value>::max()),
    INF(std::numeric_limits<Value>::has_infinity ? std::numeric_limits<Value>::infinity() : MAX)
  {
    set_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  void set_params(unsigned n_iter_max, Value epsilon_large_factor, Value epsilon_small_factor) {
    n_iter_max_ = n_iter_max;
    epsilon_large_ = epsilon_large_factor * std::numeric_limits<Value>::epsilon();
    epsilon_small_ = epsilon_small_factor * std::numeric_limits<Value>::epsilon();
  }

  // get description of this network simplex
  std::string description() const {
    std::ostringstream oss;
    oss << "  NetworkSimplex\n"
        << "    n_iter_max - " << n_iter_max_    << '\n'
        << "    epsilon_large - " << epsilon_large_ << '\n'
        << "    epsilon_small - " << epsilon_small_ << '\n';
    return oss.str();
  }

  // set dists and weights
  ValueVector & weights() { return _supplies; }
  ValueVector & dists() { return _costs; }

  // run computation given init, weights, dists
  NetworkSimplexStatus compute(std::size_t n0, std::size_t n1) {

    construct_graph(n0, n1);
    NetworkSimplexStatus status(run());

    // store total cost if network simplex had success
    if (status == NetworkSimplexStatus::Success) {
      total_cost_ = 0;
      for (Arc a = 0; a < arcNum(); a++)
        total_cost_ += _flows[a] * _costs[a];
    }
    else total_cost_ = INVALID_COST_VALUE;

    return status;
  }

  // access total cost
  Value total_cost() const { return total_cost_; }

  // flow and ground_dist vectors, only first n0_*n0_ values should be used
  const ValueVector & dists() const { return _costs; }
  const ValueVector & flows() const { return _flows; }

  // free all memory (rarely used, probably only relevant when doing massive computations)
  void free() {
    free_vector(_costs);
    free_vector(_supplies);
    free_vector(_flows);
    free_vector(_pis);
    free_vector(_sources);
    free_vector(_targets);
    free_vector(_parents);
    free_vector(_threads);
    free_vector(_rev_threads);
    free_vector(_succ_nums);
    free_vector(_last_succs);
    free_vector(_dirty_revs);
    free_vector(_preds);
    free_vector(_arc_mins);
    free_vector(_forwards);
    free_vector(_states);
  }

private:

  //---------------------------------------------------------------------------
  // Internal enums
  //---------------------------------------------------------------------------

  // State constants for arcs
  enum ArcState : char {
    STATE_UPPER = -1,
    STATE_TREE  =  0,
    STATE_LOWER =  1
  };

  //---------------------------------------------------------------------------
  // Data storage
  //---------------------------------------------------------------------------
  
  // constructor parameters
  unsigned n_iter_max_, iter_;
  Value epsilon_large_, epsilon_small_;

  // large consts initialized in constructor
  Value MAX, INF;

  // cost flow storage vectors
  ValueVector _costs; // ground distances between nodes
  ValueVector _supplies; // supply values of the nodes
  ValueVector _flows; // flow along each arc
  ValueVector _pis; // potentials of the nodes
  NodeVector _sources; // ids of the source nodes
  NodeVector _targets; // ids of the target nodes

  // spanning tree structure vectors
  NodeVector _parents, _threads, _rev_threads, _succ_nums, _last_succs, _dirty_revs;
  ArcVector _preds, _arc_mins;
  BoolVector _forwards;
  StateVector _states;

  // variables of BlockSearchPivotRule
  Arc _next_arc;
  Node _block_size;

  // other variables
  Value _sum_supplies, total_cost_;

  // Temporary data used in the current pivot iteration
  Arc in_arc;
  Node join, u_in, v_in, u_out, v_out, first, second, stem, par_stem, new_stem, right, last;
  Value delta;

  //---------------------------------------------------------------------------
  // FullBipartiteGraph functionality
  //---------------------------------------------------------------------------

  // number of nodes in each part of the graph
  Node n0_, n1_;

  // total number of nodes and arcs
  Node node_num_;
  Arc arc_num_;

  // resets internal size of the graph
  /* Nodes are indexed from 0 to nodeNum - 1
    - The first n0 correspond to the first part of the graph, indices 0, ..., n0 - 1
    - The last n1 corresponds to the second part of the graph, indices n0, ..., n0 + n1 - 1
  */
  void construct_graph(std::size_t n0, std::size_t n1) {

    n0_ = n0;
    n1_ = n1;
    node_num_ = n0_ + n1_;
    arc_num_ = Arc(n0_)*Arc(n1_);

    if (n0 + n1 > std::numeric_limits<Node>::max())
      throw std::overflow_error("Too many nodes for " + std::to_string(sizeof(Node)) + " byte Node type");
    if (n0 != 0 && arc_num_ / n0 != n1)
      throw std::overflow_error("Too many arcs for " + std::to_string(sizeof(Arc)) + " byte Arc type");
  }

  // access functions
  Node nodeNum() const { return node_num_; }
  Arc arcNum() const { return arc_num_; }
  Node nsource() const { return n0_; }
  Node ntarget() const { return n1_; }

  // max ID functions
  Node maxNodeId() const { return node_num_ - 1; }
  Arc maxArcId() const { return arc_num_ - 1; }

  // get node from arc
  Node source(Arc arc) const { return arc / n1_; }
  Node target(Arc arc) const { return (arc % n1_) + n0_; }

  // index helpers
  Node firstNode() const { return maxNodeId(); }
  Arc firstArc() const { return maxArcId(); }
  static void next(Node & node) { --node; }
  static void next(Arc & arc) { --arc; }

  // index helper for iterating over arcs entering a given target node
  void firstIn(Arc & arc, const Node t) const {
    if (t < n0_) arc = INVALID;
    else arc = arc_num_ + t - node_num_; // using the fact that target node id is t - n0_ and node_num is n0_ + n1_
  }
  void nextIn(Arc & arc) const {
    arc -= n1_;
    if (arc < 0) arc = INVALID;
  }

  //---------------------------------------------------------------------------
  // Initialization methods, called from `run`
  //---------------------------------------------------------------------------

  NetworkSimplexStatus run() {

    // reset vectors that are sized according to number of nodes
    Node all_node_num(nodeNum() + 1); // includes extra 1 for root node
    _supplies.resize(all_node_num);
    _pis.resize(all_node_num);
    _parents.resize(all_node_num);
    _threads.resize(all_node_num);
    _rev_threads.resize(all_node_num);
    _succ_nums.resize(all_node_num);
    _last_succs.resize(all_node_num);
    _preds.resize(all_node_num);
    _forwards.resize(all_node_num);

    // reset vectors sized according to number of arcs
    Arc all_arc_num(arcNum() + nodeNum()); // preparing for EQ constraints in init
    _costs.resize(all_arc_num);
    _flows.resize(all_arc_num);
    _sources.resize(all_arc_num); // look into storing this since they will be recomputed many times
    _targets.resize(all_arc_num); // look into storing this since they will be recomputed many times
    _states.resize(all_arc_num);

    // zero out flow (later nodes are initialized below)
    std::fill(_flows.begin(), _flows.begin() + arcNum(), 0);

    // store the arcs in the original order
    for (Arc a = 0; a < arcNum(); a++) {
      _sources[a] = source(a);
      _targets[a] = target(a);
    }

    // check for empty problem
    if (nodeNum() == 0) return NetworkSimplexStatus::Empty;

    // check supply total and make secondary supplies negative
    _sum_supplies = 0;
    for (Node i = 0; i < nsource(); i++)
      _sum_supplies += _supplies[i];
    for (Node i = nsource(); i < nodeNum(); i++)
      _sum_supplies += (_supplies[i] *= -1);
    if (std::fabs(_sum_supplies) > epsilon_large_) {
      std::cerr << "sum_supplies " << _sum_supplies << '\n';
      return NetworkSimplexStatus::SupplyMismatch;
    }
    _sum_supplies = 0;

    // initialize artificial cost
    Value art_costs;
    if (std::numeric_limits<Value>::is_exact)
      art_costs = std::numeric_limits<Value>::max() / 2 + 1;
    else
      art_costs = (*std::max_element(_costs.begin(), _costs.end()) + 1) * nodeNum();

    // initialize arc maps
    std::fill(_states.begin(), _states.begin() + arcNum(), STATE_LOWER);

    // set data for the artificial root node
    Node root(nodeNum());
    _parents[root] = -1;
    _preds[root] = -1;
    _threads[root] = 0;
    _rev_threads[0] = root;
    _succ_nums[root] = nodeNum() + 1;
    _last_succs[root] = root - 1;
    _supplies[root] = -_sum_supplies;
    _pis[root] = 0;

    // EQ supply constraints
    Arc e(arcNum());
    for (Node u = 0; u < nodeNum(); u++, e++) {
      _parents[u] = root;
      _preds[u] = e;
      _threads[u] = u + 1;
      _rev_threads[u + 1] = u;
      _succ_nums[u] = 1;
      _last_succs[u] = u;
      _states[e] = STATE_TREE;
      if (_supplies[u] >= 0) {
        _forwards[u] = true;
        _pis[u] = 0;
        _sources[e] = u;
        _targets[e] = root;
        _flows[e] = _supplies[u];
        _costs[e] = 0;
      } else {
        _forwards[u] = false;
        _pis[u] = art_costs;
        _sources[e] = root;
        _targets[e] = u;
        _flows[e] = -_supplies[u];
        _costs[e] = art_costs;
      }
    }

    // initialize block search pivot rule
    _next_arc = 0;
    _block_size = std::max(Node(BLOCK_SIZE_FACTOR * std::sqrt(Value(arcNum()))), MIN_BLOCK_SIZE);

    // perform heuristic initial pivots
    if (!initialPivots()) return NetworkSimplexStatus::Unbounded;

    // Execute the Network Simplex algorithm
    iter_ = 0;
    while (findEnteringArc()) {
      if (iter_++ >= n_iter_max_)
        return NetworkSimplexStatus::MaxIterReached;

      findJoinNode();
      bool change(findLeavingArc());
      if (delta >= MAX) return NetworkSimplexStatus::Unbounded;
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }

    // Check feasibility
    for (Arc e = arcNum(); e != all_arc_num; e++) {
      if (_flows[e] != 0) {
        if (std::fabs(_flows[e]) > epsilon_large_) {
          std::cerr << "Bad flow: " << _flows[e] << '\n';
          return NetworkSimplexStatus::Infeasible;
        }
        else _flows[e] = 0;
      }
    }

    return NetworkSimplexStatus::Success;
  }

  //---------------------------------------------------------------------------
  // BlockSearchPivotRule functionality
  //---------------------------------------------------------------------------

  // find next entering arc
  bool findEnteringArc() {
    Value a, c, min(0);
    Arc e(_next_arc);
    Node cnt(_block_size);
    for (Arc ind = 0; ind < arcNum(); ind++, e++) {
      if (e == arcNum()) e -= arcNum();

      c = _states[e] * (_costs[e] + _pis[_sources[e]] - _pis[_targets[e]]);
      if (c < min) {
        min = c;
        in_arc = e;
      }
      if (--cnt == 0) {
        Value pi_sources_in_arc(std::fabs(_pis[_sources[in_arc]])),
              pi_targets_in_arc(std::fabs(_pis[_targets[in_arc]])),
              cost_in_arc(std::fabs(_costs[in_arc]));
        a = pi_sources_in_arc > pi_targets_in_arc ? pi_sources_in_arc : pi_targets_in_arc;
        if (a < cost_in_arc) a = cost_in_arc;
        if (min < -epsilon_small_*a) {
          _next_arc = e;
          return true;
        }
        cnt = _block_size;
      }
    }
    Value pi_sources_in_arc(std::fabs(_pis[_sources[in_arc]])),
          pi_targets_in_arc(std::fabs(_pis[_targets[in_arc]])),
          cost_in_arc(std::fabs(_costs[in_arc]));
    a = pi_sources_in_arc > pi_targets_in_arc ? pi_sources_in_arc : pi_targets_in_arc;
    if (a < cost_in_arc) a = cost_in_arc;
    if (min < -epsilon_small_*a) {
      _next_arc = e;
      return true;
    }
    return false;
  }

  //---------------------------------------------------------------------------
  // Helper routines for running network simplex algorithm
  //---------------------------------------------------------------------------

  // Heuristic initial pivots
  bool initialPivots() {

    // Find the min. cost incomming arc for each demand node
    _arc_mins.clear();
    _arc_mins.reserve(ntarget());
    for (Node v = nsource(); v < nodeNum(); v++) {
      Value c, min_costs = std::numeric_limits<Value>::max();
      Arc a, min_arc(INVALID);
      for (firstIn(a, v); a != INVALID; nextIn(a)) {
        c = _costs[a];
        if (c < min_costs) {
          min_costs = c;
          min_arc = a;
        }
      }
      if (min_arc != INVALID)
        _arc_mins.push_back(min_arc);
    }

    // Perform heuristic initial pivots
    for (Arc a : _arc_mins) {
      in_arc = a;
      if (_states[in_arc] * (_costs[in_arc] + _pis[_sources[in_arc]] - _pis[_targets[in_arc]]) >= 0) continue;
      findJoinNode();
      bool change(findLeavingArc());
      if (delta >= MAX) return false;
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }
    return true;
  }

  // Find the join node
  void findJoinNode() {
    Node u(_sources[in_arc]), v(_targets[in_arc]);
    while (u != v) {
      if (_succ_nums[u] < _succ_nums[v]) u = _parents[u];
      else v = _parents[v];
    }
    join = u;
  }

  // Find the leaving arc of the cycle and returns true if the
  // leaving arc is not the same as the entering arc
  bool findLeavingArc() {

    // Initialize first and second nodes according to the direction of the cycle
    if (_states[in_arc] == STATE_LOWER) {
      first  = _sources[in_arc];
      second = _targets[in_arc];
    } else {
      first  = _targets[in_arc];
      second = _sources[in_arc];
    }
    delta = INF;
    char result(0);
    Value d;

    // Search the cycle along the path from the first node to the root
    for (Node u = first; u != join; u = _parents[u]) {
      d = _forwards[u] ? _flows[_preds[u]] : INF;
      if (d < delta) {
        delta = d;
        u_out = u;
        result = 1;
      }
    }
    // Search the cycle along the path form the second node to the root
    for (Node u = second; u != join; u = _parents[u]) {
      d = _forwards[u] ? INF : _flows[_preds[u]];
      if (d <= delta) {
        delta = d;
        u_out = u;
        result = 2;
      }
    }

    if (result == 1) {
      u_in = first;
      v_in = second;
    } else {
      u_in = second;
      v_in = first;
    }

    return result != 0;
  }

  // Change _flows and _states vectors
  void changeFlow(bool change) {

    // Augment along the cycle
    if (delta > 0) {
      Value val = _states[in_arc] * delta;
      _flows[in_arc] += val;
      for (Node u = _sources[in_arc]; u != join; u = _parents[u])
        _flows[_preds[u]] += _forwards[u] ? -val : val;
      for (Node u = _targets[in_arc]; u != join; u = _parents[u])
        _flows[_preds[u]] += _forwards[u] ? val : -val;
    }

    // Update the state of the entering and leaving arcs
    if (change) {
      _states[in_arc] = STATE_TREE;
      _states[_preds[u_out]] = (_flows[_preds[u_out]] == 0) ? STATE_LOWER : STATE_UPPER;
    }
    else _states[in_arc] = -_states[in_arc];
  }

  // Update the tree structure
  void updateTreeStructure() {
    Node w, u(_last_succs[u_in]), old_rev_threads(_rev_threads[u_out]), 
         old_succ_nums(_succ_nums[u_out]), old_last_succs(_last_succs[u_out]);
    v_out = _parents[u_out];
    right = _threads[u];    // the node after it

    // Handle the case when old_rev_threads equals to v_in (it also means that join and v_out coincide)
    if (old_rev_threads == v_in) last = _threads[_last_succs[u_out]];
    else last = _threads[v_in];

    // Update _threads and _parents along the stem nodes (i.e. the nodes
    // between u_in and u_out, whose parent have to be changed)
    _threads[v_in] = stem = u_in;
    _dirty_revs.clear();
    _dirty_revs.push_back(v_in);
    par_stem = v_in;
    while (stem != u_out) {

      // Insert the next stem node into the thread list
      new_stem = _parents[stem];
      _threads[u] = new_stem;
      _dirty_revs.push_back(u);

      // Remove the subtree of stem from the thread list
      w = _rev_threads[stem];
      _threads[w] = right;
      _rev_threads[right] = w;

      // Change the parent node and shift stem nodes
      _parents[stem] = par_stem;
      par_stem = stem;
      stem = new_stem;

      // Update u and right
      u = _last_succs[stem] == _last_succs[par_stem] ? _rev_threads[par_stem] : _last_succs[stem];
      right = _threads[u];
    }
    _parents[u_out] = par_stem;
    _threads[u] = last;
    _rev_threads[last] = _last_succs[u_out] = u;

    // Remove the subtree of u_out from the thread list except for
    // the case when old_rev_threads equals to v_in
    // (it also means that join and v_out coincide)
    if (old_rev_threads != v_in) {
      _threads[old_rev_threads] = right;
      _rev_threads[right] = old_rev_threads;
    }

    // Update _rev_threads using the new _threads values
    for (Node u : _dirty_revs)
      _rev_threads[_threads[u]] = u;

    // Update _preds, _forwards, _last_succs and _succ_nums for the
    // stem nodes from u_out to u_in
    Node tmp_sc(0), tmp_ls(_last_succs[u_out]);
    u = u_out;
    while (u != u_in) {
      w = _parents[u];
      _preds[u] = _preds[w];
      _forwards[u] = !_forwards[w];
      tmp_sc += _succ_nums[u] - _succ_nums[w];
      _succ_nums[u] = tmp_sc;
      _last_succs[w] = tmp_ls;
      u = w;
    }
    _preds[u_in] = in_arc;
    _forwards[u_in] = (u_in == _sources[in_arc]);
    _succ_nums[u_in] = old_succ_nums;

    // Set limits for updating _last_succs from v_in and v_out towards the root
    Node up_limit_in(-1), up_limit_out(-1);
    if (_last_succs[join] == v_in) up_limit_out = join;
    else up_limit_in = join;

    // Update _last_succs from v_in towards the root
    for (u = v_in; u != up_limit_in && _last_succs[u] == v_in; u = _parents[u])
      _last_succs[u] = _last_succs[u_out];

    // Update _last_succs from v_out towards the root
    if (join != old_rev_threads && v_in != old_rev_threads)
      for (u = v_out; u != up_limit_out && _last_succs[u] == old_last_succs; u = _parents[u])
        _last_succs[u] = old_rev_threads;
    else 
      for (u = v_out; u != up_limit_out && _last_succs[u] == old_last_succs; u = _parents[u])
        _last_succs[u] = _last_succs[u_out];

    // Update _succ_nums from v_in to join
    for (u = v_in; u != join; u = _parents[u])
      _succ_nums[u] += old_succ_nums;

    // Update _succ_nums from v_out to join
    for (u = v_out; u != join; u = _parents[u])
      _succ_nums[u] -= old_succ_nums;
  }

  // Update potentials
  void updatePotential() {
    Value sigma = _forwards[u_in] ? _pis[v_in] - _pis[u_in] - _costs[_preds[u_in]] : _pis[v_in] - _pis[u_in] + _costs[_preds[u_in]];

    // Update potentials in the subtree, which has been moved
    Node end = _threads[_last_succs[u_in]];
    for (Node u = u_in; u != end; u = _threads[u])
      _pis[u] += sigma;
  }

}; // NetworkSimplex

} // namespace lemon

#endif // LEMON_NETWORK_SIMPLEX_HH
