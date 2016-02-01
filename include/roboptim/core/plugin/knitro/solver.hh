// Copyright (C) 2014 by Thomas Moulard, AIST, CNRS.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HH
# define ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HH

# include <string>
# include <ostream>
# include <map>

# include <knitro.h>

# include <roboptim/core/portability.hh>
# include <roboptim/core/differentiable-function.hh>
# include <roboptim/core/linear-function.hh>
# include <roboptim/core/quadratic-function.hh>
# include <roboptim/core/solver.hh>
# include <roboptim/core/util.hh>

namespace roboptim
{
  template <typename T> class KNITROSolver;

  /// \brief KNITRO iteration callback.
  /// \tparam T matrix type.
  template <typename T>
  int iterationCallback (KTR_context_ptr kc, const int n, const int m,
                         const int nnzJ, const double* const x,
                         const double* const lambda, const double obj,
                         const double* const c, const double* const objGrad,
                         const double* const jac, void* userParams);

  /// \brief Compute callback.
  /// \tparam T matrix type.
  template <typename T>
  int computeCallback (const int evalRequestCode, const int n, const int m,
                       const int nnzJ, const int nnzH, const double* const x,
                       const double* const lambda, double* const obj,
                       double* const c, double* const objGrad,
                       double* const jac, double* const hessian,
                       double* const hessVector, void* userParams);

  /// \brief Helper to compute the objective's gradient.
  /// \tparam T matrix type.
  /// \param objGrad raw buffer to the objective's gradient.
  /// \param x argument vector.
  /// \param solver KNITRO solver.
  template <typename T>
  void evalGradObj (double* const objGrad,
                    typename GenericFunction<T>::const_argument_ref x,
                    const KNITROSolver<T>* solver);

  /// \brief Helper to compute blocks of the Jacobian matrix.
  /// \tparam T matrix type.
  /// \param jac raw buffer to the Jacobian matrix.
  /// \param x argument vector.
  /// \param solver KNITRO solver.
  template <typename T>
  void evalJacobian (double* const jac,
                     typename GenericFunction<T>::const_argument_ref x,
                     const KNITROSolver<T>* solver);

  /// \addtogroup roboptim_solver
  /// @{

  /// \brief KNITRO based solver.
  /// \tparam T matrix type.
  template <typename T>
  class ROBOPTIM_DLLEXPORT KNITROSolver : public Solver<T>
  {

  public:
    /// \brief Parent type.
    typedef Solver<T> solver_t;

    typedef typename solver_t::problem_t problem_t;
    typedef typename solver_t::callback_t callback_t;
    typedef typename solver_t::solverState_t solverState_t;
    typedef typename solver_t::parameters_t parameters_t;

    typedef GenericDifferentiableFunction<T> differentiableFunction_t;
    typedef GenericLinearFunction<T> linearFunction_t;
    typedef GenericQuadraticFunction<T> quadraticFunction_t;

    typedef typename problem_t::function_t function_t;
    typedef typename function_t::matrix_t matrix_t;
    typedef typename function_t::value_type value_type;
    typedef typename function_t::size_type size_type;
    typedef typename function_t::vector_t vector_t;
    typedef typename function_t::argument_t argument_t;
    typedef typename function_t::result_t result_t;
    typedef typename differentiableFunction_t::gradient_t gradient_t;
    typedef typename differentiableFunction_t::jacobian_t jacobian_t;

    /// \brief Constructor.
    /// \param problem problem that will be solved.
    explicit KNITROSolver (const problem_t& problem);

    /// \brief Destructor.
    virtual ~KNITROSolver ();

    /// \brief Solve the problem.
    virtual void solve ();

    /// \brief Initialize KNITRO parameters.
    void initializeParameters ();

    /// \brief Update KNITRO parameters.
    void updateParameters ();

    /// \brief Display the solver on the specified output stream.
    ///
    /// \param o output stream used for display
    /// \return output stream
    virtual std::ostream& print (std::ostream& o) const;

    /// \brief Set the user-defined iteration callback.
    /// \param callback iteration callback.
    void setIterationCallback (callback_t callback);

    /// \brief Get the user callback.
    const callback_t& callback () const;

    /// \brief Get the current solver state.
    solverState_t& solverState () const;

    /// \brief Output size of the problem.
    size_type outputSize () const;

    /// \brief Input size of the problem.
    size_type inputSize () const;

    /// \brief Get the buffer for the gradient of the objective function.
    gradient_t& objGrad () const;

  private:
    /// \brief Create the log directory.
    /// KNITRO does not create it if it does not exist.
    void createLogDir () const;

    /// \brief Fill result data structure.
    /// \tparam R result type.
    /// \param res result data structure.
    /// \param x argument vector at the optimal solution.
    /// \param lambda lambda at the optimal solution.
    /// \param obj objective at the optimal solution.
    template <typename R>
    void fillResult (R& res, const argument_t& x, const vector_t& lambda,
                     const result_t& obj) const;

    /// \brief Initialize the KNITRO solver.
    /// This will loop until a proper license is obtained.
    void initializeKnitro ();

    /// \brief Get the sparsity pattern of the Jacobian matrix.
    /// \param jacIndexVars indices for variables (columns).
    /// \param jacIndexCons indices for constraints (rows).
    /// \param n input size.
    /// \param m output size.
    /// \param x initial x (used only in the sparse case).
    /// \return number of nonzeros in the Jacobian matrix.
    int getSparsityPattern (Eigen::VectorXi& jacIndexVars,
                            Eigen::VectorXi& jacIndexCons, int n, int m,
                            const argument_t& x) const;

    /// \brief Get the starting point, or generate a proper one.
    /// \return starting point.
    argument_t initialArgument () const;

  private:
    /// \brief Input size.
    size_type n_;

    /// \brief Output size.
    size_type m_;

    /// \brief Per-iteration callback.
    callback_t callback_;

    /// \brief Current state of the solver (used by the callback function).
    mutable solverState_t solverState_;

    /// \brief Objective gradient buffer.
    mutable gradient_t objGrad_;

    /// \brief Map from strings to KNITRO enums.
    std::map<std::string, int> stringToEnum_;

    /// \brief Time to wait between 2 retries of the KNITRO initialization, in
    /// milliseconds.
    /// This is useful when sharing a single network license...
    unsigned waitTime_;

    /// \brief KNITRO solver context.
    KTR_context* knitro_;
  };

  /// @}

} // end of namespace roboptim

# include "roboptim/core/plugin/knitro/solver.hxx"

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HH
