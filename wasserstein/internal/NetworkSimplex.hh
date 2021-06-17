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

#ifndef WASSERSTEIN_NETWORK_SIMPLEX_HH
#define WASSERSTEIN_NETWORK_SIMPLEX_HH

// C++ standard library
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

//-----------------------------------------------------------------------------
// NetworkSimplex
//-----------------------------------------------------------------------------

namespace {

// main parameters of block search pivot rule, consider changing these to check for speed
const double BLOCK_SIZE_FACTOR = 1.0;
const int MIN_BLOCK_SIZE = 10;
const int INVALID = -1;
const double INVALID_COST_VALUE = -1.0;

}

// templated NetworkSimplex class
// - Value: floating point type that is used for computations
// - Node: signed integer type that indexes particles
// - Arc: signed integer type that (roughly) can hold the product of two Nodes
// - Bool: boolean type (often not "bool" to avoid std::vector<bool> being slow)
template<typename V, typename A, typename N, typename B>
class NetworkSimplex {

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & n_iter_max_ & epsilon_large_ & epsilon_small_;
  }
#endif

public:

  // EMD-style typedefs
  typedef V value_type;

  // old-style typedefs
  typedef N Node;
  typedef A Arc;
  typedef V Value;
  typedef B Bool;

  // rough type checking
  static_assert(std::is_integral<Node>::value && std::is_signed<Node>::value,
                "Node should be a signed integral type.");
  static_assert(std::is_integral<Arc>::value && std::is_signed<Arc>::value,
                "Arc should be a signed integral type.");
  static_assert(sizeof(Arc) >= sizeof(Node), "Arc type should be bigger-than-or-equal-to Node type");
  static_assert(std::is_floating_point<Value>::value, "Value should be a floating point type.");
  static_assert(std::is_integral<Bool>::value, "Bool should be an integral type.");

  // vector typedefs
  typedef std::vector<Node> NodeVector;
  typedef std::vector<Arc> ArcVector;
  typedef std::vector<Value> ValueVector;
  typedef std::vector<Bool> BoolVector;
  typedef std::vector<char> StateVector;

  // default constructor
  NetworkSimplex() :
    MAX(std::numeric_limits<Value>::max()),
    INF(std::numeric_limits<Value>::has_infinity ? std::numeric_limits<Value>::infinity() : MAX)
  {}

  // constructor
  NetworkSimplex(std::size_t n_iter_max, Value epsilon_large_factor, Value epsilon_small_factor) :
    NetworkSimplex()
  {
    set_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  void set_params(std::size_t n_iter_max, Value epsilon_large_factor, Value epsilon_small_factor) {
    n_iter_max_ = n_iter_max;
    epsilon_large_ = epsilon_large_factor * std::numeric_limits<Value>::epsilon();
    epsilon_small_ = epsilon_small_factor * std::numeric_limits<Value>::epsilon();
  }

  // get description of this network simplex
  std::string description() const {
    std::ostringstream oss;
    oss << "  NetworkSimplex\n"
        << "    n_iter_max - "    << n_iter_max_    << '\n'
        << "    epsilon_large - " << epsilon_large_ << '\n'
        << "    epsilon_small - " << epsilon_small_ << '\n';
    return oss.str();
  }

  // set dists and weights
  ValueVector & weights() { return supplies_; }
  ValueVector & dists() { return costs_; }

  // run computation given init, weights, dists
  EMDStatus compute(std::size_t n0, std::size_t n1) {

    construct_graph(n0, n1);
    EMDStatus status(run());

    // store total cost if network simplex had success
    if (status == EMDStatus::Success) {
      total_cost_ = 0;
      for (Arc a = 0; a < arcNum(); a++)
        total_cost_ += flows_[a] * costs_[a];
    }
    else total_cost_ = INVALID_COST_VALUE;

    return status;
  }

  // access total cost
  Value total_cost() const { return total_cost_; }

  // access number of iterations required
  std::size_t n_iter() const { return n_iter_; }

  // flow and ground_dist vectors, only first n0_*n0_ values should be used
  const ValueVector & dists() const { return costs_; }
  const ValueVector & flows() const { return flows_; }
  const ValueVector & potentials() const { return pis_; }

  // free all memory (rarely used, probably only relevant when doing massive computations)
  void free() {
    free_vector(costs_);
    free_vector(supplies_);
    free_vector(flows_);
    free_vector(pis_);
    free_vector(sources_);
    free_vector(targets_);
    free_vector(parents_);
    free_vector(threads_);
    free_vector(rev_threads_);
    free_vector(succ_nums_);
    free_vector(last_succs_);
    free_vector(dirty_revs_);
    free_vector(preds_);
    free_vector(arc_mins_);
    free_vector(forwards_);
    free_vector(states_);
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
  std::size_t n_iter_max_, n_iter_;
  Value epsilon_large_, epsilon_small_;

  // large consts initialized in constructor
  Value MAX, INF;

  // cost flow storage vectors
  ValueVector costs_; // ground distances between nodes
  ValueVector flows_; // flow along each arc
  ValueVector supplies_; // supply values of the nodes
  ValueVector pis_; // potentials of the nodes
  NodeVector sources_; // ids of the source nodes
  NodeVector targets_; // ids of the target nodes

  // spanning tree structure vectors
  NodeVector parents_, threads_, rev_threads_, succ_nums_, last_succs_, dirty_revs_;
  ArcVector preds_, arc_mins_;
  BoolVector forwards_;
  StateVector states_;

  // variables of BlockSearchPivotRule
  Arc next_arc_;
  Node block_size_;

  // other variables
  Value sum_supplies_, total_cost_;

  // Temporary data used in the current pivot iteration
  Arc in_arc_;
  Node join_, u_in_, v_in_, u_out_, v_out_;
  Value delta_;

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
  template<typename I>
  static void next(I & arc) { --arc; }

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

  EMDStatus run() {

    // reset vectors that are sized according to number of nodes
    Node all_node_num(nodeNum() + 1); // includes extra 1 for root node
    supplies_.resize(all_node_num);
    pis_.resize(all_node_num);
    parents_.resize(all_node_num);
    threads_.resize(all_node_num);
    rev_threads_.resize(all_node_num);
    succ_nums_.resize(all_node_num);
    last_succs_.resize(all_node_num);
    preds_.resize(all_node_num);
    forwards_.resize(all_node_num);

    // reset vectors sized according to number of arcs
    Arc all_arc_num(arcNum() + nodeNum()); // preparing for EQ constraints in init
    costs_.resize(all_arc_num);
    flows_.resize(all_arc_num);
    sources_.resize(all_arc_num); // look into storing this since they will be recomputed many times
    targets_.resize(all_arc_num); // look into storing this since they will be recomputed many times
    states_.resize(all_arc_num);

    // zero out flow (later nodes are initialized below)
    std::fill(flows_.begin(), flows_.begin() + arcNum(), 0);

    // store the arcs in the original order
    for (Arc a = 0; a < arcNum(); a++) {
      sources_[a] = source(a);
      targets_[a] = target(a);
    }

    // check for empty problem
    if (nodeNum() == 0) return EMDStatus::Empty;

    // check supply total and make secondary supplies negative
    sum_supplies_ = 0;
    for (Node i = 0; i < nsource(); i++)
      sum_supplies_ += supplies_[i];
    for (Node i = nsource(); i < nodeNum(); i++)
      sum_supplies_ += (supplies_[i] *= -1);
    if (std::fabs(sum_supplies_) > epsilon_large_) {
      std::cerr << "sumsupplies_ " << sum_supplies_ << '\n';
      return EMDStatus::SupplyMismatch;
    }
    sum_supplies_ = 0;

    // initialize artificial cost
    Value artcosts;
    if (std::numeric_limits<Value>::is_exact)
      artcosts = std::numeric_limits<Value>::max() / 2 + 1;
    else
      artcosts = (*std::max_element(costs_.begin(), costs_.end()) + 1) * nodeNum();

    // initialize arc maps
    std::fill(states_.begin(), states_.begin() + arcNum(), STATE_LOWER);

    // set data for the artificial root node
    Node root(nodeNum());
    parents_[root] = -1;
    preds_[root] = -1;
    threads_[root] = 0;
    rev_threads_[0] = root;
    succ_nums_[root] = nodeNum() + 1;
    last_succs_[root] = root - 1;
    supplies_[root] = -sum_supplies_;
    pis_[root] = 0;

    // EQ supply constraints
    Arc e(arcNum());
    for (Node u = 0; u < nodeNum(); u++, e++) {
      parents_[u] = root;
      preds_[u] = e;
      threads_[u] = u + 1;
      rev_threads_[u + 1] = u;
      succ_nums_[u] = 1;
      last_succs_[u] = u;
      states_[e] = STATE_TREE;
      if (supplies_[u] >= 0) {
        forwards_[u] = true;
        pis_[u] = 0;
        sources_[e] = u;
        targets_[e] = root;
        flows_[e] = supplies_[u];
        costs_[e] = 0;
      } else {
        forwards_[u] = false;
        pis_[u] = artcosts;
        sources_[e] = root;
        targets_[e] = u;
        flows_[e] = -supplies_[u];
        costs_[e] = artcosts;
      }
    }

    // initialize block search pivot rule
    next_arc_ = 0;
    block_size_ = std::max(Node(BLOCK_SIZE_FACTOR * std::sqrt(Value(arcNum()))), MIN_BLOCK_SIZE);

    // perform heuristic initial pivots
    if (!initialPivots()) return EMDStatus::Unbounded;

    // Execute the Network Simplex algorithm
    n_iter_ = 0;
    while (findEnteringArc()) {
      if (n_iter_++ >= n_iter_max_)
        return EMDStatus::MaxIterReached;

      findJoinNode();
      bool change(findLeavingArc());
      if (delta_ >= MAX) return EMDStatus::Unbounded;
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }

    // Check feasibility
    for (Arc e = arcNum(); e != all_arc_num; e++) {
      if (flows_[e] != 0) {
        if (std::fabs(flows_[e]) > epsilon_large_) {
          std::cerr << "Bad flow: " << flows_[e] << '\n';
          return EMDStatus::Infeasible;
        }
        else flows_[e] = 0;
      }
    }

    return EMDStatus::Success;
  }

  //---------------------------------------------------------------------------
  // BlockSearchPivotRule functionality
  //---------------------------------------------------------------------------

  // find next entering arc
  bool findEnteringArc() {
    Value a, c, min(0);
    Arc e(next_arc_);
    Node cnt(block_size_);
    for (Arc ind = 0; ind < arcNum(); ind++, e++) {
      if (e == arcNum()) e -= arcNum();

      c = states_[e] * (costs_[e] + pis_[sources_[e]] - pis_[targets_[e]]);
      if (c < min) {
        min = c;
        in_arc_ = e;
      }
      if (--cnt == 0) {
        Value pisources__in_arc_(std::fabs(pis_[sources_[in_arc_]])),
              pitargets__in_arc_(std::fabs(pis_[targets_[in_arc_]])),
              cost_in_arc_(std::fabs(costs_[in_arc_]));
        a = pisources__in_arc_ > pitargets__in_arc_ ? pisources__in_arc_ : pitargets__in_arc_;
        if (a < cost_in_arc_) a = cost_in_arc_;
        if (min < -epsilon_small_*a) {
          next_arc_ = e;
          return true;
        }
        cnt = block_size_;
      }
    }
    Value pisources__in_arc_(std::fabs(pis_[sources_[in_arc_]])),
          pitargets__in_arc_(std::fabs(pis_[targets_[in_arc_]])),
          cost_in_arc_(std::fabs(costs_[in_arc_]));
    a = pisources__in_arc_ > pitargets__in_arc_ ? pisources__in_arc_ : pitargets__in_arc_;
    if (a < cost_in_arc_) a = cost_in_arc_;
    if (min < -epsilon_small_*a) {
      next_arc_ = e;
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
    arc_mins_.clear();
    arc_mins_.reserve(ntarget());
    for (Node v = nsource(); v < nodeNum(); v++) {
      Value c, mincosts_ = std::numeric_limits<Value>::max();
      Arc a, min_arc_(INVALID);
      for (firstIn(a, v); a != INVALID; nextIn(a)) {
        c = costs_[a];
        if (c < mincosts_) {
          mincosts_ = c;
          min_arc_ = a;
        }
      }
      if (min_arc_ != INVALID)
        arc_mins_.push_back(min_arc_);
    }

    // Perform heuristic initial pivots
    for (Arc a : arc_mins_) {
      in_arc_ = a;
      if (states_[in_arc_] * (costs_[in_arc_] + pis_[sources_[in_arc_]] - pis_[targets_[in_arc_]]) >= 0) continue;
      findJoinNode();
      bool change(findLeavingArc());
      if (delta_ >= MAX) return false;
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }
    return true;
  }

  // Find the join_ node
  void findJoinNode() {
    Node u(sources_[in_arc_]), v(targets_[in_arc_]);
    while (u != v) {
      if (succ_nums_[u] < succ_nums_[v]) u = parents_[u];
      else v = parents_[v];
    }
    join_ = u;
  }

  // Find the leaving arc of the cycle and returns true if the
  // leaving arc is not the same as the entering arc
  bool findLeavingArc() {

    // Initialize first and second nodes according to the direction of the cycle
    Node first, second;
    if (states_[in_arc_] == STATE_LOWER) {
      first  = sources_[in_arc_];
      second = targets_[in_arc_];
    } else {
      first  = targets_[in_arc_];
      second = sources_[in_arc_];
    }

    delta_ = INF;
    char result(0);
    Value d;

    // Search the cycle along the path from the first node to the root
    for (Node u = first; u != join_; u = parents_[u]) {
      d = forwards_[u] ? flows_[preds_[u]] : INF;
      if (d < delta_) {
        delta_ = d;
        u_out_ = u;
        result = 1;
      }
    }

    // Search the cycle along the path form the second node to the root
    for (Node u = second; u != join_; u = parents_[u]) {
      d = forwards_[u] ? INF : flows_[preds_[u]];
      if (d <= delta_) {
        delta_ = d;
        u_out_ = u;
        result = 2;
      }
    }

    if (result == 1) {
      u_in_ = first;
      v_in_ = second;
    } else {
      u_in_ = second;
      v_in_ = first;
    }

    return result != 0;
  }

  // Change flows_ and states_ vectors
  void changeFlow(bool change) {

    // Augment along the cycle
    if (delta_ > 0) {
      Value val = states_[in_arc_] * delta_;
      flows_[in_arc_] += val;
      for (Node u = sources_[in_arc_]; u != join_; u = parents_[u])
        flows_[preds_[u]] += forwards_[u] ? -val : val;
      for (Node u = targets_[in_arc_]; u != join_; u = parents_[u])
        flows_[preds_[u]] += forwards_[u] ? val : -val;
    }

    // Update the state of the entering and leaving arcs
    if (change) {
      states_[in_arc_] = STATE_TREE;
      states_[preds_[u_out_]] = (flows_[preds_[u_out_]] == 0) ? STATE_LOWER : STATE_UPPER;
    }
    else states_[in_arc_] = -states_[in_arc_];
  }

  // Update the tree structure
  void updateTreeStructure() {
    Node w, u(last_succs_[u_in_]), oldrev_threads_(rev_threads_[u_out_]), 
         oldsucc_nums_(succ_nums_[u_out_]), oldlast_succs_(last_succs_[u_out_]),
         right(threads_[u]), stem(u_in_), par_stem(v_in_), new_stem, last;
    v_out_ = parents_[u_out_];

    // Handle the case when oldrev_threads_ equals to v_in_ (it also means that join_ and v_out_ coincide)
    if (oldrev_threads_ == v_in_) last = threads_[last_succs_[u_out_]];
    else last = threads_[v_in_];

    // Update threads_ and parents_ along the stem nodes (i.e. the nodes
    // between u_in_ and u_out_, whose parent have to be changed)
    threads_[v_in_] = u_in_;
    dirty_revs_.clear();
    dirty_revs_.push_back(v_in_);
    while (stem != u_out_) {

      // Insert the next stem node into the thread list
      new_stem = parents_[stem];
      threads_[u] = new_stem;
      dirty_revs_.push_back(u);

      // Remove the subtree of stem from the thread list
      w = rev_threads_[stem];
      threads_[w] = right;
      rev_threads_[right] = w;

      // Change the parent node and shift stem nodes
      parents_[stem] = par_stem;
      par_stem = stem;
      stem = new_stem;

      // Update u and right
      u = last_succs_[stem] == last_succs_[par_stem] ? rev_threads_[par_stem] : last_succs_[stem];
      right = threads_[u];
    }
    parents_[u_out_] = par_stem;
    threads_[u] = last;
    rev_threads_[last] = last_succs_[u_out_] = u;

    // Remove the subtree of u_out_ from the thread list except for
    // the case when oldrev_threads_ equals to v_in_
    // (it also means that join_ and v_out_ coincide)
    if (oldrev_threads_ != v_in_) {
      threads_[oldrev_threads_] = right;
      rev_threads_[right] = oldrev_threads_;
    }

    // Update rev_threads_ using the new threads_ values
    for (Node u : dirty_revs_)
      rev_threads_[threads_[u]] = u;

    // Update preds_, forwards_, last_succs_ and succ_nums_ for the
    // stem nodes from u_out_ to u_in_
    Node tmp_sc(0), tmp_ls(last_succs_[u_out_]);
    u = u_out_;
    while (u != u_in_) {
      w = parents_[u];
      preds_[u] = preds_[w];
      forwards_[u] = !forwards_[w];
      tmp_sc += succ_nums_[u] - succ_nums_[w];
      succ_nums_[u] = tmp_sc;
      last_succs_[w] = tmp_ls;
      u = w;
    }
    preds_[u_in_] = in_arc_;
    forwards_[u_in_] = (u_in_ == sources_[in_arc_]);
    succ_nums_[u_in_] = oldsucc_nums_;

    // Set limits for updating last_succs_ from v_in_ and v_out_ towards the root
    Node up_limit_in(-1), up_limit_out(-1);
    if (last_succs_[join_] == v_in_) up_limit_out = join_;
    else up_limit_in = join_;

    // Update last_succs_ from v_in_ towards the root
    for (u = v_in_; u != up_limit_in && last_succs_[u] == v_in_; u = parents_[u])
      last_succs_[u] = last_succs_[u_out_];

    // Update last_succs_ from v_out_ towards the root
    if (join_ != oldrev_threads_ && v_in_ != oldrev_threads_)
      for (u = v_out_; u != up_limit_out && last_succs_[u] == oldlast_succs_; u = parents_[u])
        last_succs_[u] = oldrev_threads_;
    else 
      for (u = v_out_; u != up_limit_out && last_succs_[u] == oldlast_succs_; u = parents_[u])
        last_succs_[u] = last_succs_[u_out_];

    // Update succ_nums_ from v_in_ to join_
    for (u = v_in_; u != join_; u = parents_[u])
      succ_nums_[u] += oldsucc_nums_;

    // Update succ_nums_ from v_out_ to join_
    for (u = v_out_; u != join_; u = parents_[u])
      succ_nums_[u] -= oldsucc_nums_;
  }

  // Update potentials
  void updatePotential() {
    Value sigma = forwards_[u_in_] ? pis_[v_in_] - pis_[u_in_] - costs_[preds_[u_in_]] : pis_[v_in_] - pis_[u_in_] + costs_[preds_[u_in_]];

    // Update potentials in the subtree, which has been moved
    Node end = threads_[last_succs_[u_in_]];
    for (Node u = u_in_; u != end; u = threads_[u])
      pis_[u] += sigma;
  }

}; // NetworkSimplex

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_NETWORK_SIMPLEX_HH
