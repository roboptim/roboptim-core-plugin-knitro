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
# include <boost/filesystem.hpp>
# include <boost/static_assert.hpp>

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
    typedef typename solver_t::vector_t vector_t;
    typedef typename solver_t::solverState_t solverState_t;

    if (!userParams)
    {
      std::cerr << "bad user params\n";
      return -1;
    }
    const solver_t* solver = static_cast<const solver_t*> (userParams);

    if (!solver->callback ()) return 0;

    solverState_t& solverState = solver->solverState ();
    solverState.x () = Eigen::Map<const vector_t> (x, n);
    solverState.cost () = obj;
    solverState.constraintViolation () = KTR_get_abs_feas_error (kc);

    // call user-defined callback
    solver->callback () (solver->problem (), solverState);

    return 0;
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
    typedef typename solver_t::gradient_t gradient_t;
    typedef typename solver_t::jacobian_t jacobian_t;
    typedef typename solver_t::result_t result_t;
    typedef
      typename solver_t::problem_t::constraints_t::const_iterator iterator_t;
    typedef
      typename solver_t::differentiableFunction_t differentiableFunction_t;

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
        constraintsBuf.segment (idx, (*it)->outputSize ()) = (*(*it)) (x_);
        idx += (*it)->outputSize ();
      }
    }
    // evaluate cost gradient and constraints jacobian
    else if (evalRequestCode == KTR_RC_EVALGA)
    {
      Eigen::Map<gradient_t> objGrad_ (objGrad, n);

      // cost gradient
      const differentiableFunction_t* df = 0;
      if (solver->problem ()
            .function ()
            .template asType<differentiableFunction_t> ())
      {
        df = solver->problem ().function ().castInto<DifferentiableFunction> ();
        objGrad_ = df->gradient (x_);
      }

      // constraints jacobian
      ptrdiff_t idx = 0;
      df = 0;
      Eigen::Map<jacobian_t> jacobianBuf (jac, m, n);
      for (iterator_t it = solver->problem ().constraints ().begin ();
           it != solver->problem ().constraints ().end (); ++it)
      {
        if ((*it)->template asType<differentiableFunction_t> ())
        {
          df = (*it)->castInto<differentiableFunction_t> ();
          df->jacobian (jacobianBuf.block (idx, 0, (*it)->outputSize (), n),
                        x_);
          idx += (*it)->outputSize ();
        }
      }
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
    : solver_t (problem), solverState_ (problem), knitro_ ()
  {
    initializeParameters ();
  }

  template <typename T>
  KNITROSolver<T>::~KNITROSolver ()
  {
    KTR_free (&knitro_);
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

    // KNITRO specific.
    // Much more options are available for Knitro see Knitro documentation

    if (!knitro_) knitro_ = KTR_new ();
    if (!knitro_) throw std::runtime_error ("failed to initialize KNITRO");

    if (KTR_set_func_callback (knitro_, &computeCallback<T>))
      throw std::runtime_error ("failed to set evaluation callback");
    if (KTR_set_grad_callback (knitro_, &computeCallback<T>))
      throw std::runtime_error ("failed to set gradient callback");

    //  Output
    DEFINE_PARAMETER ("knitro.outlev", "output verbosity level", 4);
    DEFINE_PARAMETER ("knitro.outmode",
                      "output style (\"screen\", \"file\" or \"both\")",
                      std::string ("file"));
    DEFINE_PARAMETER ("knitro.outdir", "output directory", std::string ("."));
    DEFINE_PARAMETER ("knitro.outappend", "output append mode", false);

    // Gradient and hessian used
    DEFINE_PARAMETER ("knitro.gradopt", "type of gradient method used",
                      std::string ("GRADOPT_FORWARD"));
    DEFINE_PARAMETER ("knitro.hessopt", "type of hessian method used",
                      std::string ("HESSOPT_BFGS"));

    //  Termination
    DEFINE_PARAMETER ("knitro.maxit", "maximum number of iteration permitted",
                      3000);
    DEFINE_PARAMETER ("knitro.opttol",
                      "desired convergence tolerance (relative)", 1e-6);
    DEFINE_PARAMETER ("knitro.feastol", "desired threshold for the feasibility",
                      1e-6);
    DEFINE_PARAMETER ("knitro.xtol",
                      "desired threshold for the constraint violation", 1e-5);

    //  Barrier parameter
    DEFINE_PARAMETER ("knitro.bar_initmu", "barrier initial mu", 1e-1);

    // Algorithm choice.
    DEFINE_PARAMETER ("knitro.algorithm", "type of solver algorithm",
                      std::string ("ALG_BAR_DIRECT"));

    // Miscellaneous
    DEFINE_PARAMETER ("knitro.par_numthreads",
                      "number of parallel threads to use", 1);

    stringToEnum_ = knitroParameterMap ();
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
    int n = static_cast<int> (pb.function ().inputSize ());

    // number of constraints
    int m = static_cast<int> (pb.constraintsOutputSize ());
    int nnzJ = n * m;
    int nnzH = 0;

    int objType = KTR_OBJTYPE_GENERAL;
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
        // FIXME: dispatch linear constraints here
        cType[offset + j] = KTR_CONTYPE_GENERAL;
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
    argument_t xInitial (n);
    if (pb.startingPoint ())
      xInitial = *pb.startingPoint ();
    else
      xInitial.setZero ();

    // sparsity pattern (dense here)
    Eigen::VectorXi jacIndexVars (nnzJ);
    Eigen::VectorXi jacIndexCons (nnzJ);

    // FIXME: this depends on RowMajor/ColMajor
    BOOST_STATIC_ASSERT (Eigen::ROBOPTIM_STORAGE_ORDER == Eigen::ColMajor);
    int k = 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
      {
        jacIndexCons[k] = j;
        jacIndexVars[k] = i;
        k++;
      }

    nStatus =
      KTR_init_problem (knitro_, n, objGoal, objType, xLoBnds.data (),
                        xUpBnds.data (), m, cType.data (), cLoBnds.data (),
                        cUpBnds.data (), nnzJ, jacIndexVars.data (),
                        jacIndexCons.data (), nnzH, 0, 0, xInitial.data (), 0);

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

    KTR_free (&knitro_);
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
    if (logmode != "file" and logmode != "both") return;

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
    size_type n = x.size ();
    size_type m = lambda.size () - n;

    res.x = x;
    res.lambda.resize (n + m);
    res.lambda.segment (0, n) = lambda.segment (m, n);
    if (m > 0)
    {
      res.lambda.segment (n, m) = lambda.segment (0, m);
      res.constraints.resize (m);
      KTR_get_constraint_values (knitro_, res.constraints.data ());
    }
    res.value = obj;
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
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_SOLVER_HXX
