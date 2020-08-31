/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


#include <mpi.h>
#include <cstring>
#include <cmath>
#include "compute_solvation.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "kspace.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

enum{OFF,INTER,INTRA};

/* ---------------------------------------------------------------------- */

ComputeSolvation::ComputeSolvation(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  group2(NULL)
{
  if (narg < 5) error->all(FLERR,"Illegal compute solvation command");

  scalar_flag = vector_flag = 1;
  size_vector = 3;
  extscalar = 1;
  extvector = 1;

  cutoff = atof(arg[4])*atof(arg[4]);

  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2,arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute solvation group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  pairflag = 1;
  kspaceflag = 0;
  boundaryflag = 1;
  molflag = OFF;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute solvation command");
      if (strcmp(arg[iarg+1],"yes") == 0) pairflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) pairflag = 0;
      else error->all(FLERR,"Illegal compute solvation command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute solvation command");
      if (strcmp(arg[iarg+1],"yes") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) kspaceflag = 0;
      else error->all(FLERR,"Illegal compute solvation command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute solvation command");
      if (strcmp(arg[iarg+1],"yes") == 0) boundaryflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) boundaryflag  = 0;
      else error->all(FLERR,"Illegal compute solvation command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"molecule") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute solvation command");
      if (strcmp(arg[iarg+1],"off") == 0) molflag = OFF;
      else if (strcmp(arg[iarg+1],"inter") == 0) molflag = INTER;
      else if (strcmp(arg[iarg+1],"intra") == 0) molflag  = INTRA;
      else error->all(FLERR,"Illegal compute solvation command");
      if (molflag != OFF && atom->molecule_flag == 0)
        error->all(FLERR,"Compute solvation molecule requires molecule IDs");
      iarg += 2;
    } else error->all(FLERR,"Illegal compute solvation command");
  }

  vector = new double[3];
}

/* ---------------------------------------------------------------------- */

ComputeSolvation::~ComputeSolvation()
{
  delete [] group2;
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeSolvation::init()
{
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0

  if (pairflag && force->pair == NULL)
    error->all(FLERR,"No pair style defined for compute group/group");
  if (force->pair_match("hybrid",0) == NULL && force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support compute group/group");

  // error if Kspace style does not compute group/group interactions

  if (kspaceflag && force->kspace == NULL)
    error->all(FLERR,"No Kspace style defined for compute group/group");
  if (kspaceflag && force->kspace->group_group_enable == 0)
    error->all(FLERR,"Kspace style does not support compute group/group");

  if (pairflag) {
    pair = force->pair;
    cutsq = force->pair->cutsq;
  } else pair = NULL;


  // recheck that group 2 has not been deleted

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute group/group group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  // need an occasional half neighbor list

  if (pairflag) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeSolvation::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

double ComputeSolvation::compute_scalar()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;

  if (pairflag) pair_contribution();
  if (kspaceflag) kspace_contribution();

  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeSolvation::compute_vector()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;

  if (pairflag) pair_contribution();
  if (kspaceflag) kspace_contribution();
}

/* ---------------------------------------------------------------------- */

void ComputeSolvation::pair_contribution()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,eng,fpair,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  // invoke half neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I,J are not in 2 groups

  double one[4];
  one[0] = one[1] = one[2] = one[3] = 0.0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    // skip if atom I is not in either group
    if (!(mask[i] & groupbit || mask[i] & jgroupbit)) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      // skip if atom J is not in either group

      if (!(mask[j] & groupbit || mask[j] & jgroupbit)) continue;

      // skip if atoms I,J are only in the same group

      int ij_flag = 0;
      int ji_flag = 0;
      if (mask[i] & groupbit && mask[j] & jgroupbit) ij_flag = 1;
      if (mask[j] & groupbit && mask[i] & jgroupbit) ji_flag = 1;
      if (!ij_flag && !ji_flag) continue;

      // skip if molecule IDs of atoms I,J do not satisfy molflag setting

      if (molflag != OFF) {
        if (molflag == INTER) {
          if (molecule[i] == molecule[j]) continue;
        } else {
          if (molecule[i] != molecule[j]) continue;
        }
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutoff) {
        eng = pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);

        // energy only computed once so tally full amount
        // force tally is jgroup acting on igroup

        if (newton_pair || j < nlocal) {
          one[0] += 0.5*eng;
          if (ij_flag) {
            one[1] += delx*fpair;
            one[2] += dely*fpair;
            one[3] += delz*fpair;
          }
          if (ji_flag) {
            one[1] -= delx*fpair;
            one[2] -= dely*fpair;
            one[3] -= delz*fpair;
          }

        // energy computed twice so tally half amount
        // only tally force if I own igroup atom

        } else {
          one[0] += 0.0*eng;
          if (ij_flag) {
            one[1] += delx*fpair;
            one[2] += dely*fpair;
            one[3] += delz*fpair;
          }
        }
      }
    }
  }

  double all[4];
  MPI_Allreduce(one,all,4,MPI_DOUBLE,MPI_SUM,world);
  scalar += all[0];
  vector[0] += all[1]; vector[1] += all[2]; vector[2] += all[3];
}

