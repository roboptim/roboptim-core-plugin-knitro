// Copyright (C) 2016 by Benjamin Chrétien CNRS-AIST JRL.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation either version 3 of the License or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not see <http://www.gnu.org/licenses/>.

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH
# define ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH

# include <string>
# include <map>
# include <cstdio>

# include <boost/assign/list_of.hpp>
# include <boost/preprocessor/stringize.hpp>

# include <knitro.h>

namespace roboptim
{
  /// \brief Ignore standard output (cout or cerr).
  class IgnoreStream
  {
  public:
    IgnoreStream (FILE* file) : file_ (file), fd_ (-1)
    {
    }

    ~IgnoreStream ()
    {
      end ();
    }

    /// \brief Start redirection.
    void start ()
    {
      if (file_ && fd_ < 0)
      {
        // Redirect to /dev/null
        fd_ = dup (fileno (file_));
        freopen ("/dev/null", "w", file_);
      }
    }

    /// \brief End redirection.
    void end ()
    {
      if (fd_ >= 0)
      {
        // Redirect back to initial output
        fclose (file_);
        FILE* fbk = fdopen (fd_, "w");
        *file_ = *fbk;
        fd_ = -1;
      }
    }

  private:
    FILE* file_;
    int fd_;
  };

  /// \brief Return a map from the string representation of KNITRO options to
  /// their actual value (enums).
  /// That way, users can provide the option as a string, rather than needing to
  /// include knitro.h to set the options.
  std::map<std::string, int> knitroParameterMap ()
  {
#define REGISTER_OPTION(OPT) \
    (BOOST_PP_STRINGIZE(OPT), KTR_ ## OPT)

    // Load <algo string algo> map
    // Note: the full list contains over 300 options. We stick to the main ones
    // for now.
    std::map<std::string, int> stringToEnum = boost::assign::map_list_of
      REGISTER_OPTION(ALG_ACTIVE)
      REGISTER_OPTION(ALG_ACT_CG)
      REGISTER_OPTION(ALG_AUTO)
      REGISTER_OPTION(ALG_AUTOMATIC)
      REGISTER_OPTION(ALG_BAR_CG)
      REGISTER_OPTION(ALG_BAR_DIRECT)
      REGISTER_OPTION(ALG_IPCG)
      REGISTER_OPTION(ALG_IPDIRECT)
      REGISTER_OPTION(ALG_MULTI)
      REGISTER_OPTION(GRADOPT_CENTRAL)
      REGISTER_OPTION(GRADOPT_EXACT)
      REGISTER_OPTION(GRADOPT_FORWARD)
      REGISTER_OPTION(HESSIAN_NO_F_ALLOW)
      REGISTER_OPTION(HESSIAN_NO_F_FORBID)
      REGISTER_OPTION(HESSOPT_BFGS)
      REGISTER_OPTION(HESSOPT_EXACT)
      REGISTER_OPTION(HESSOPT_FINITE_DIFF)
      REGISTER_OPTION(HESSOPT_LBFGS)
      REGISTER_OPTION(HESSOPT_PRODUCT)
      REGISTER_OPTION(HESSOPT_SR1)
      REGISTER_OPTION(LINSOLVER_AUTO)
      REGISTER_OPTION(LINSOLVER_DENSEQR)
      REGISTER_OPTION(LINSOLVER_HYBRID)
      REGISTER_OPTION(LINSOLVER_INTERNAL)
      REGISTER_OPTION(LINSOLVER_MA27)
      REGISTER_OPTION(LINSOLVER_MA57);

#undef REGISTER_OPTION

    return stringToEnum;
  }

#define MAP_KNITRO_ERRORS(MACRO)                                               \
  MACRO (KTR_RC_FEAS_XTOL,                                                     \
         "Primal feasible solution; the optimization terminated because the "  \
         "relative change in the solution estimate is less than that "         \
         "specified by the parameter xtol. To try to get more accuracy one "   \
         "may decrease xtol. If xtol is very small already, it is an "         \
         "indication that no more significant progress can be made. It's "     \
         "possible the approximate feasible solution is optimal, but perhaps " \
         "the stopping tests cannot be satisfied because of degeneracy, "      \
         "ill-conditioning or bad scaling.");                                  \
  MACRO (KTR_RC_FEAS_NO_IMPROVE,                                               \
         "Primal feasible solution estimate cannot be improved; desired "      \
         "accuracy in dual feasibility could not be achieved. No further "     \
         "progress can be made. It's possible the approximate feasible "       \
         "solution is optimal, but perhaps the stopping tests cannot be "      \
         "satisfied because of degeneracy, ill-conditioning or bad scaling."); \
  MACRO (KTR_RC_FEAS_FTOL,                                                     \
         "Primal feasible solution; the optimization terminated because the "  \
         "relative change in the objective function is less than that "        \
         "specified by the parameter ftol for ftol_iters consecutive "         \
         "iterations. To try to get more accuracy one may decrease ftol "      \
         "and/or increase ftol_iters. If ftol is very small already, it is "   \
         "an indication that no more significant progress can be made. It's "  \
         "possible the approximate feasible solution is optimal, but perhaps " \
         "the stopping tests cannot be satisfied because of degeneracy, "      \
         "ill-conditioning or bad scaling.");                                  \
  MACRO (KTR_RC_INFEASIBLE,                                                    \
         "Convergence to an infeasible point. Problem may be locally "         \
         "infeasible. If problem is believed to be feasible, try multistart "  \
         "to search for feasible points. The algorithm has converged to an "   \
         "infeasible point from which it cannot further decrease the "         \
         "infeasibility measure. This happens when the problem is "            \
         "infeasible, but may also occur on occasion for feasible problems "   \
         "with nonlinear constraints or badly scaled problems. It is "         \
         "recommended to try various initial points with the multi-start "     \
         "feature. If this occurs for a variety of initial points, it is "     \
         "likely the problem is infeasible.");                                 \
  MACRO (KTR_RC_INFEAS_XTOL,                                                   \
         "Terminate at infeasible point because the relative change in the "   \
         "solution estimate is less than that specified by the parameter "     \
         "xtol. To try to find a feasible point one may decrease xtol. If "    \
         "xtol is very small already, it is an indication that no more "       \
         "significant progress can be made. It is recommended to try various " \
         "initial points with the multi-start feature. If this occurs for a "  \
         "variety of initial points, it is likely the problem is "             \
         "infeasible.");                                                       \
  MACRO (KTR_RC_INFEAS_NO_IMPROVE,                                             \
         "Current infeasible solution estimate cannot be improved. Problem "   \
         "may be badly scaled or perhaps infeasible. If problem is believed "  \
         "to be feasible, try multistart to search for feasible points. If "   \
         "this occurs for a variety of initial points, it is likely the "      \
         "problem is infeasible.");                                            \
  MACRO (KTR_RC_INFEAS_MULTISTART,                                             \
         "Multistart: no primal feasible point found. The multi-start "        \
         "feature was unable to find a feasible point. If the problem is "     \
         "believed to be feasible, then increase the number of initial "       \
         "points tried in the multi-start feature and also perhaps increase "  \
         "the range from which random initial points are chosen.");            \
  MACRO (KTR_RC_INFEAS_CON_BOUNDS,                                             \
         "The constraint bounds have been determined to be infeasible.");      \
  MACRO (KTR_RC_INFEAS_VAR_BOUNDS,                                             \
         "The variable bounds have been determined to be infeasible.");        \
  MACRO (KTR_RC_UNBOUNDED,                                                     \
         "Problem appears to be unbounded. Iterate is feasible and objective " \
         "magnitude is greater than objrange. The objective function appears " \
         "to be decreasing without bound, while satisfying the constraints. "  \
         "If the problem really is bounded, increase the size of the "         \
         "parameter objrange to avoid terminating with this message.");        \
  MACRO (KTR_RC_CALLBACK_ERR,                                                  \
         "Callback function error. This termination value indicates that an "  \
         "error (i.e., negative return value) occurred in a user provided "    \
         "callback routine.");                                                 \
  MACRO (KTR_RC_LP_SOLVER_ERR,                                                 \
         "LP solver error. This termination value indicates that an "          \
         "unrecoverable error occurred in the LP solver used in the "          \
         "active-set algorithm preventing the optimization from continuing."); \
  MACRO (KTR_RC_EVAL_ERR,                                                      \
         "Evaluation error. This termination value indicates that an "         \
         "evaluation error occurred (e.g., divide by 0, taking the square "    \
         "root of a negative number), preventing the optimization from "       \
         "continuing.");                                                       \
  MACRO (KTR_RC_OUT_OF_MEMORY,                                                 \
         "Not enough memory available to solve problem. This termination "     \
         "value indicates that there was not enough memory available to "      \
         "solve the problem.")

#define MAP_KNITRO_WARNINGS(MACRO)                                            \
  MACRO (KTR_RC_NEAR_OPT,                                                     \
         "iPrimal feasible solution estimate cannot be improved. It appears " \
         "to be optimal, but desired accuracy in dual feasibility could not " \
         "be achieved. No more progress can be made, but the stopping tests " \
         "are close to being satisfied (within a factor of 100) and so the "  \
         "current approximate solution is believed to be optimal.");          \
  MACRO (KTR_RC_ITER_LIMIT,                                                   \
         "The iteration limit was reached before being able to satisfy the "  \
         "required stopping criteria. A feasible point was found. The "       \
         "iteration limit can be increased through the user option maxit.");  \
  MACRO (KTR_RC_TIME_LIMIT,                                                   \
         "The time limit was reached before being able to satisfy the "       \
         "required stopping criteria. A feasible point was found. The time "  \
         "limit can be increased through the user options maxtime_cpu and "   \
         "maxtime_real.");                                                    \
  MACRO (KTR_RC_USER_TERMINATION, "Knitro has been terminated by the user.")

  // TODO: investigate the following error code:
  // MACRO (KTR_RC_FEVAL_LIMIT,
  //        "The function evaluation limit was reached before being able to "
  //        "satisfy the required stopping criteria. A feasible point was "
  //        "found. The function evaluation limit can be increased through the "
  //        "user option maxfevals.");

} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH
