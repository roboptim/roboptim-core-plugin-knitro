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

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_SOLVER_HH
# define ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_SOLVER_HH

# include <iostream>
# include <string>
# include <utility>
# include <vector>

# include <boost/mpl/vector.hpp>
# include <boost/variant.hpp>
# include <boost/optional.hpp>

# include <roboptim/core/portability.hh>
# include <roboptim/core/differentiable-function.hh>
# include <roboptim/core/derivable-function.hh>
# include <roboptim/core/linear-function.hh>
# include <roboptim/core/solver.hh>
# include <roboptim/core/util.hh>

/// \brief KNITRO Context pre-declaration.
class KTR_context;

namespace roboptim
{
  /// \addtogroup roboptim_solver
  /// @{

  /// \brief KNITRO based solver.
  class ROBOPTIM_DLLEXPORT KNITROSolver : public Solver<EigenMatrixDense>
  {

  public:
    typedef DifferentiableFunction differentiableFunction_t;

    typedef problem_t::function_t function_t;
    typedef function_t::matrix_t matrix_t;
    typedef function_t::value_type value_type;
    typedef function_t::vector_t vector_t;
    typedef function_t::argument_t argument_t;
    typedef function_t::result_t result_t;
    typedef differentiableFunction_t::gradient_t gradient_t;
    typedef differentiableFunction_t::jacobian_t jacobian_t;

    /// \brief Parent type.
    typedef Solver<EigenMatrixDense> parent_t;

    /// \param problem problem that will be solved
    explicit KNITROSolver (const problem_t& problem);

    virtual ~KNITROSolver ();

    /// \brief Solve the problem.
    virtual void solve ();

    void initializeParameters ();

    void updateParameters ();

    /// \brief Display the solver on the specified output stream.
    ///
    /// \param o output stream used for display
    /// \return output stream
    virtual std::ostream& print (std::ostream& o) const;

    void setIterationCallback (callback_t callback);

    const callback_t& callback () const
    {
      return callback_;
    }

    solverState_t& solverState () const
    {
      return solverState_;
    }

  private:
    /// \brief Per-iteration callback.
    callback_t callback_;

    /// \brief Current state of the solver (used by the callback function).
    mutable solverState_t solverState_;

    /// \brief KNITRO solver context.
    KTR_context* knitro_;
  };

  /// @}

} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_SOLVER_HH
