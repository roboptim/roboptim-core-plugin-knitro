// Copyright (C) 2014 by Thomas Moulard, AIST, CNRS.
// Copyright (C) 2016 by Benjamin Chrétien, CNRS-AIST JRL.
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

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HXX
# define ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HXX

# include <string>
# include <ostream>
# include <stdexcept>

# include <boost/variant.hpp>
# include <boost/scoped_ptr.hpp>
# include <boost/filesystem.hpp>
# include <boost/date_time/posix_time/posix_time.hpp>
# include <boost/thread/thread.hpp>

# include <Eigen/Core>

# include <knitro.h>

# include <roboptim/core/result.hh>
# include <roboptim/core/result-with-warnings.hh>
# include <roboptim/core/function.hh>
# include <roboptim/core/util.hh>

# include "roboptim/core/plugin/knitro/util.hh"
# include "roboptim/core/plugin/knitro/parameters-updater.hh"

namespace roboptim
{
  template <typename T>
  int iterationCallback (KTR_context_ptr kc, const int n, const int /* m */,
                         const int /* nnzJ */, const double* const x,
                         const double* const /* lambda */, const double obj,
                         const double* const /* c */,
                         const double* const /* objGrad */,
                         const double* const /* jac */, void* userParams)
  {
    typedef KNITROSolver<T> solver_t;
    typedef typename solver_t::argument_t argument_t;
    typedef typename solver_t::solverState_t solverState_t;

    if (!userParams)
    {
      std::cerr << "bad user params\n";
      return -1;
    }
    const solver_t* solver = static_cast<const solver_t*> (userParams);

    if (!solver->callback ()) return 0;

    solverState_t& solverState = solver->solverState ();
    const Eigen::Map<const argument_t> map_x (x, n);
    solverState.x () = map_x;

    // First iteration does not provide cost and constraint violation
    int iter = KTR_get_number_iters (kc);
    if (iter > 0)
    {
      solverState.cost () = obj;
      solverState.constraintViolation () = KTR_get_abs_feas_error (kc);
    }
    else
    {
      // Need to evaluate cost and violation for the callback
      solverState.cost () = solver->problem ().function () (map_x)[0];
      solverState.constraintViolation () =
        solver->problem ().template constraintsViolation<Eigen::Infinity> (map_x);
    }

    // Initialize value for knitro.stop
    bool stop_optim = false;
    solverState.parameters ()["knitro.stop"].value = stop_optim;

    // call user-defined callback
    solver->callback () (solver->problem (), solverState);

    // knitro.stop may have been unintentionally removed in the callback
    try
    {
      stop_optim = solverState.template getParameter<bool> ("knitro.stop");
    }
    catch (std::out_of_range&)
    {
      stop_optim = false;
    }

    // Terminate if asked by the user
    return stop_optim? KTR_RC_USER_TERMINATION : 0;
  }

  template <typename T>
  int computeCallback (const int evalRequestCode, const int n, const int m,
                       const int /* nnzJ */, const int /* nnzH */,
                       const double* const x, const double* const /* lambda */,
                       double* const obj, double* const c,
                       double* const objGrad, double* const jac,
                       double* const /* hessian */,
                       double* const /* hessVector */, void* userParams)
  {
    typedef KNITROSolver<T> solver_t;
    typedef typename solver_t::argument_t argument_t;
    typedef typename solver_t::result_t result_t;
    typedef
      typename solver_t::problem_t::constraints_t::const_iterator iterator_t;

    if (!userParams)
    {
      std::cerr << "bad user params\n";
      return -1;
    }
    solver_t* solver = static_cast<solver_t*> (userParams);

    const Eigen::Map<const argument_t> x_ (x, n);

    // ask to evaluate objective and constraints
    if (evalRequestCode == KTR_RC_EVALFC)
    {
      Eigen::Map<result_t> obj_ (obj, 1);

      // objective
      obj_ = solver->problem ().function () (x_);

      // constraints
      ptrdiff_t idx = 0;
      Eigen::Map<result_t> constraintsBuf (c, m);
      for (iterator_t it = solver->problem ().constraints ().begin ();
           it != solver->problem ().constraints ().end (); ++it)
      {
        (*(*it)) (constraintsBuf.segment (idx, (*it)->outputSize ()), x_);
        idx += (*it)->outputSize ();
      }
    }
    // evaluate cost gradient and constraints jacobian
    else if (evalRequestCode == KTR_RC_EVALGA)
    {
      // objective gradient
      evalGradObj (objGrad, x_, solver);

      // constraints jacobian
      evalJacobian (jac, x_, solver);
    }
    else
    {
      std::cerr << "bad evalRequestCode\n";
      return -1;
    }
    return 0;
  }

  template <typename T>
  KNITROSolver<T>::KNITROSolver (const problem_t& problem)
    : solver_t (problem),
      n_ (problem.function ().inputSize ()),
      m_ (problem.constraintsOutputSize ()),
      solverState_ (problem),
      objGrad_ (n_),
      jac_ (),
      waitTime_ (1000),
      maxRetries_ (300),
      knitro_ (0),
      zlm_ (0)
  {
    initializeKnitro ();
    initializeParameters ();
  }

  template <typename T>
  KNITROSolver<T>::~KNITROSolver ()
  {
    KTR_free (&knitro_);
    ZLM_release_license (zlm_);
  }

  template <typename T>
  void KNITROSolver<T>::initializeKnitro ()
  {
    zlm_ = ZLM_checkout_license ();

    IgnoreStream ignoreCerr (stderr);
    unsigned ntry = 0;
    while (!knitro_ && ntry < maxRetries_)
    {
      // Try to obtain a license
      // Note: we ignore error messages to avoid being spammed by KNITRO
      if (ntry > 0) ignoreCerr.start ();
      knitro_ = KTR_new_zlm (NULL, NULL, zlm_);
      if (ntry > 0) ignoreCerr.end ();

      if (!knitro_)
      {
        if (ntry == 0)
        {
          std::cerr << "Waiting for KNITRO license...";
        }
        else
        {
          std::cerr << ".";
        }
        ntry++;
        boost::this_thread::sleep (boost::posix_time::milliseconds (waitTime_));
      }
    }
    if (ntry > 0)
      std::cerr << std::endl;
    if (ntry == maxRetries_)
      throw std::runtime_error ("could not obtain a KNITRO license");
  }

#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)     \
  do                                                  \
  {                                                   \
    this->parameters_[KEY].description = DESCRIPTION; \
    this->parameters_[KEY].value = VALUE;             \
  } while (0)

  template <typename T>
  void KNITROSolver<T>::initializeParameters ()
  {
    this->parameters_.clear ();

    if (KTR_set_func_callback (knitro_, &computeCallback<T>))
      throw std::runtime_error ("failed to set evaluation callback");
    if (KTR_set_grad_callback (knitro_, &computeCallback<T>))
      throw std::runtime_error ("failed to set gradient callback");

    // Much more options are available for Knitro see Knitro documentation
    //  Output
    DEFINE_PARAMETER ("knitro.outlev", "output verbosity level", 4);
    DEFINE_PARAMETER ("knitro.outmode",
                      "output style (\"screen\", \"file\" or \"both\")",
                      std::string ("file"));
    DEFINE_PARAMETER ("knitro.outdir", "output directory", std::string ("."));
    DEFINE_PARAMETER ("knitro.outappend", "output append mode", false);

    // Gradient and hessian used
    DEFINE_PARAMETER ("knitro.gradopt", "type of gradient method used",
                      std::string ("GRADOPT_EXACT"));
    DEFINE_PARAMETER ("knitro.hessopt", "type of hessian method used",
                      std::string ("HESSOPT_BFGS"));

    //  Termination
    DEFINE_PARAMETER ("knitro.opttol",
                      "desired convergence tolerance (relative)", 1e-6);
    DEFINE_PARAMETER ("knitro.feastol", "desired threshold for the feasibility",
                      1e-6);
    DEFINE_PARAMETER ("knitro.xtol", "tolerance on arguments", 1e-15);

    //  Barrier parameter
    DEFINE_PARAMETER ("knitro.bar_initmu", "barrier initial mu", 1e-1);

    // Algorithm choice.
    DEFINE_PARAMETER ("knitro.algorithm", "type of solver algorithm",
                      std::string ("ALG_AUTOMATIC"));

    // Miscellaneous
    DEFINE_PARAMETER ("knitro.par_numthreads",
                      "number of parallel threads to use", 1);
    DEFINE_PARAMETER ("knitro.datacheck",
                      "whether to perform more extensive data checks to look "
                      "for errors in the problem input to Knitro",
                      false);

    // Shared parameters.
    DEFINE_PARAMETER ("max-iterations", "maximum number of iterations", 3000);

    stringToEnum_ = knitroParameterMap ();

    // Initialize solver state
    solverState_.parameters ()["knitro.stop"].value = false;
    solverState_.parameters ()["knitro.stop"].description =
      "whether to stop the optimization process";
  }

  template <typename T>
  void KNITROSolver<T>::updateParameters ()
  {
    // Create the log directory
    createLogDir ();

    // If a user-defined callback was set
    if (callback ())
    {
      DEFINE_PARAMETER ("knitro.newpoint",
                        "action on a newpoint (e.g. callback)",
                        std::string ("user"));

      if (KTR_set_newpt_callback (knitro_, &iterationCallback<T>))
        throw std::runtime_error ("failed to set solver callback");
    }

    // Remap standardized parameters.
    boost::apply_visitor
      (KnitroParametersUpdater
       (knitro_, "maxit", stringToEnum_), this->parameters_["max-iterations"].value);

    // KNITRO parameters
    const std::string prefix = "knitro.";
    typedef const std::pair<const std::string, Parameter> const_iterator_t;
    BOOST_FOREACH (const_iterator_t& it, this->parameters_)
    {
      if (it.first.substr (0, prefix.size ()) == prefix)
      {
        boost::apply_visitor (
          KnitroParametersUpdater (knitro_, it.first.substr (prefix.size ()),
                                   stringToEnum_),
          it.second.value);
      }
    }
  }

#undef DEFINE_PARAMETER

#define SWITCH_ERROR(NAME, ERROR)             \
  case NAME:                                  \
  {                                           \
    Result res (n, 1);                        \
    fillResult (res, x, lambda, obj);         \
    this->result_ = SolverError (ERROR, res); \
  }                                           \
  break

#define SWITCH_WARNING(NAME, WARNING)                 \
  case NAME:                                          \
  {                                                   \
    ResultWithWarnings res (n, 1);                    \
    fillResult (res, x, lambda, obj);                 \
    res.warnings.push_back (SolverWarning (WARNING)); \
    this->result_ = res;                              \
  }                                                   \
  break

  template <typename T>
  void KNITROSolver<T>::solve ()
  {
    int nStatus = 0;

    updateParameters ();

    // Problem definition
    const problem_t& pb = this->problem ();

    // number of variables
    int n = static_cast<int> (n_);

    // number of constraints
    int m = static_cast<int> (m_);

    int objType = KTR_OBJTYPE_GENERAL;
    if (pb.function ().template asType<linearFunction_t> ())
      objType = KTR_OBJTYPE_LINEAR;
    else if (pb.function ().template asType<quadraticFunction_t> ())
      objType = KTR_OBJTYPE_QUADRATIC;

    int objGoal = KTR_OBJGOAL_MINIMIZE;

    // bounds and constraints type
    vector_t xLoBnds (n);
    vector_t xUpBnds (n);
    for (int i = 0; i < n; i++)
    {
      if (pb.argumentBounds ()[i].first == -Function::infinity ())
        xLoBnds[i] = -KTR_INFBOUND;
      else
        xLoBnds[i] = pb.argumentBounds ()[i].first;

      if (pb.argumentBounds ()[i].second == Function::infinity ())
        xUpBnds[i] = KTR_INFBOUND;
      else
        xUpBnds[i] = pb.argumentBounds ()[i].second;
    }

    Eigen::VectorXi cType (m);
    vector_t cLoBnds (m);
    vector_t cUpBnds (m);

    typedef typename KNITROSolver<T>::problem_t::constraints_t::const_iterator
      iterator_t;
    ptrdiff_t offset = 0;
    for (iterator_t it = pb.constraints ().begin ();
         it != pb.constraints ().end (); ++it)
    {
      assert (offset < m);
      ptrdiff_t i = it - pb.constraints ().begin ();

      for (int j = 0; j < (*it)->outputSize (); j++)
      {
        cType[offset + j] = KTR_CONTYPE_GENERAL;
        if ((*it)->template asType<linearFunction_t> ())
          objType = KTR_CONTYPE_LINEAR;
        else if ((*it)->template asType<quadraticFunction_t> ())
          objType = KTR_CONTYPE_QUADRATIC;

        if (pb.boundsVector ()[i][j].first == -Function::infinity ())
          cLoBnds[offset + j] = -KTR_INFBOUND;
        else
          cLoBnds[offset + j] = pb.boundsVector ()[i][j].first;

        if (pb.boundsVector ()[i][j].second == Function::infinity ())
          cUpBnds[offset + j] = KTR_INFBOUND;
        else
          cUpBnds[offset + j] = pb.boundsVector ()[i][j].second;
      }

      offset += (*it)->outputSize ();
    }

    // initial point
    argument_t xInitial = initialArgument ();

    // sparsity pattern
    Eigen::VectorXi jacIndexVars;
    Eigen::VectorXi jacIndexCons;
    int nnzJ =
      getSparsityPattern (jacIndexVars, jacIndexCons, n, m, xInitial, jac_);
    int nnzH = 0;

    nStatus =
      KTR_init_problem (knitro_, n, objGoal, objType, xLoBnds.data (),
                        xUpBnds.data (), m, cType.data (), cLoBnds.data (),
                        cUpBnds.data (), nnzJ, jacIndexVars.data (),
                        jacIndexCons.data (), nnzH, 0, 0, xInitial.data (), 0);

    if (callback ())
    {
      // Initial iterate does not trigger KTR_set_newpt_callback.
      // Also, the objective is not yet available.
      iterationCallback<T> (knitro_, n, m, nnzJ, xInitial.data (), 0, 0, 0, 0,
                            0, (void*)this);
    }

    argument_t x (n);
    vector_t lambda (m + n);
    result_t obj (1);
    nStatus = KTR_solve (knitro_, x.data (), lambda.data (), 0, obj.data (), 0,
                         0, 0, 0, 0, this);

    switch (nStatus)
    {
#ifdef KTR_RC_OPTIMAL_OR_SATISFACTORY
      case KTR_RC_OPTIMAL_OR_SATISFACTORY:
#else  //! KTR_RC_OPTIMAL_OR_SATISFACTORY
      case KTR_RC_OPTIMAL:
#endif //! KTR_RC_OPTIMAL_OR_SATISFACTORY
      {
        Result res (n, 1);
        fillResult (res, x, lambda, obj);
        this->result_ = res;
        break;
      }

        MAP_KNITRO_WARNINGS (SWITCH_WARNING);
        MAP_KNITRO_ERRORS (SWITCH_ERROR);

      default:
        throw std::runtime_error ("unknown KNITRO return code");
    }
  }

#undef SWITCH_ERROR
#undef SWITCH_WARNING

  template <typename T>
  void KNITROSolver<T>::createLogDir () const
  {
    const parameters_t& params = this->parameters_;

    // If the output level is not 0
    typename parameters_t::const_iterator it_lvl, it_file, it_name;
    it_lvl = params.find ("knitro.outlev");
    if (it_lvl == params.end ()) return;

    int loglvl = boost::get<int> (it_lvl->second.value);
    if (loglvl <= 0) return;

    // If logging to file is enabled
    it_file = params.find ("knitro.outmode");
    if (it_file == params.end ()) return;

    const std::string& logmode =
      boost::get<std::string> (it_file->second.value);
    if (logmode != "file" && logmode != "both") return;

    it_name = params.find ("knitro.outdir");
    if (it_name != params.end ())
    {
      boost::filesystem::path logdir (
        boost::get<std::string> (it_name->second.value));
      if (not boost::filesystem::is_directory (logdir))
      {
        try
        {
          boost::filesystem::create_directory (logdir);
        }
        catch (const boost::filesystem::filesystem_error& e)
        {
          std::cerr << "Error when creating log directory. " << e.what ()
                    << std::endl;
        }
      }
    }
  }

  template <typename T>
  template <typename R>
  void KNITROSolver<T>::fillResult (R& res, const argument_t& x,
                                    const vector_t& lambda,
                                    const result_t& obj) const
  {
    res.x = x;
    res.lambda.resize (n_ + m_);
    res.lambda.segment (0, n_) = lambda.segment (m_, n_);
    if (m_ > 0)
    {
      res.lambda.segment (n_, m_) = lambda.segment (0, m_);
      res.constraints.resize (m_);
      KTR_get_constraint_values (knitro_, res.constraints.data ());
    }
    res.value = obj;
  }

  template <typename T>
  typename KNITROSolver<T>::argument_t KNITROSolver<T>::initialArgument () const
  {
    const problem_t& pb = this->problem ();
    argument_t x (n_);

    if (pb.startingPoint ())
      x = *pb.startingPoint ();
    else
    {
      for (typename vector_t::Index i = 0; i < n_; ++i)
      {
        // if constraint is in an interval, evaluate at middle.
        if (pb.argumentBounds ()[i].first != -Function::infinity () &&
            pb.argumentBounds ()[i].second != Function::infinity ())
          x[i] =
            (pb.argumentBounds ()[i].second - pb.argumentBounds ()[i].first) /
            2.;
        // otherwise use the non-infinite bound, or 0
        else if (pb.argumentBounds ()[i].first != -Function::infinity ())
          x[i] = pb.argumentBounds ()[i].first;
        else if (pb.argumentBounds ()[i].second != Function::infinity ())
          x[i] = pb.argumentBounds ()[i].second;
        else
          x[i] = 0.;
      }
    }

    return x;
  }

  template <typename T>
  std::ostream& KNITROSolver<T>::print (std::ostream& o) const
  {
    solver_t::print (o);
    return o;
  }

  template <typename T>
  void KNITROSolver<T>::setIterationCallback (callback_t callback)
  {
    callback_ = callback;
  }

  template <typename T>
  const typename KNITROSolver<T>::callback_t& KNITROSolver<T>::callback () const
  {
    return callback_;
  }

  template <typename T>
  typename KNITROSolver<T>::solverState_t& KNITROSolver<T>::solverState () const
  {
    return solverState_;
  }

  template <typename T>
  typename KNITROSolver<T>::gradient_t& KNITROSolver<T>::objGrad () const
  {
    return objGrad_;
  }

  template <typename T>
  typename KNITROSolver<T>::jacobian_t& KNITROSolver<T>::jacobian () const
  {
    return jac_;
  }

  template <typename T>
  typename KNITROSolver<T>::size_type KNITROSolver<T>::inputSize () const
  {
    return n_;
  }

  template <typename T>
  typename KNITROSolver<T>::size_type KNITROSolver<T>::outputSize () const
  {
    return m_;
  }
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HXX
