/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include <iostream>
#include <string>
#include <complex>
#include "so3.h"

#include "mliap_descriptor_so3.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "mliap_data.h"
#include "pair_mliap.h"


#include <cmath>
#include <cstring>

using namespace std;
using namespace LAMMPS_NS;

#define MAXLINE 1024
#define MAXWORD 3

/* ---------------------------------------------------------------------- */

MLIAPDescriptorSO3::MLIAPDescriptorSO3(LAMMPS *lmp, char *paramfilename):
  MLIAPDescriptor(lmp)
{



  cout <<"kkk in MLIAPDescriptorSo3"<< endl;

  nelements = 0;
  elements = nullptr;
  radelem = nullptr;
  wjelem = nullptr;
  so3ptr = nullptr;
  read_paramfile(paramfilename);

/*
  snaptr = new SNA(lmp, rfac0, twojmax,
                   rmin0, switchflag, bzeroflag,
                   chemflag, bnormflag, wselfallflag, nelements);
*/
  so3ptr = new SO3(lmp,10.0,lmax,nmax);

  cout <<"kkk in MLIAPDescriptorSo3 coeff "<<so3ptr->ncoeff<< endl;

  ndescriptors = so3ptr->ncoeff;


 // ndescriptors = snaptr->ncoeff;

}

/* ---------------------------------------------------------------------- */

MLIAPDescriptorSO3::~MLIAPDescriptorSO3()
{



  delete so3ptr;

}

void MLIAPDescriptorSO3::read_paramfile(char *paramfilename)
{

  cout << "kkk in MLIAPDescriptorSO3::read_paramfile" << endl;

  // set flags for required keywords

  int rcutfacflag = 0;
//  int twojmaxflag = 0;
  int nelementsflag = 0;
  int elementsflag = 0;
  int radelemflag = 0;
  int wjelemflag = 0;
  int nmaxflag=0;
  int lmaxflag=0;
  int alphaflag=0;

  // Set defaults for optional keywords

  rfac0 = 0.99363;
  rmin0 = 0.0;
  switchflag = 1;
  bzeroflag = 1;
  chemflag = 0;
  bnormflag = 0;
  wselfallflag = 0;

  // open SNAP parameter file on proc 0

  FILE *fpparam;
  if (comm->me == 0) {
    fpparam = utils::open_potential(paramfilename,lmp,nullptr);
    if (fpparam == nullptr)
      error->one(FLERR,fmt::format("Cannot open SNAP parameter file {}: {}",
                                   paramfilename, utils::getsyserror()));
  }

  char line[MAXLINE],*ptr;
  int eof = 0;
  int n,nwords;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpparam);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpparam);
      } else n = strlen(line) + 1;
    }

    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = utils::count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line
    // strip single and double quotes from words

    char* keywd = strtok(line,"' \t\n\r\f");
    char* keyval = strtok(nullptr,"' \t\n\r\f");

    if (comm->me == 0) {
      utils::logmesg(lmp, fmt::format("SNAP keyword {} {} \n", keywd, keyval));
    }

    // check for keywords with one value per element

    if (strcmp(keywd,"elems") == 0 ||
        strcmp(keywd,"radelems") == 0 ||
        strcmp(keywd,"welems") == 0) {

      if (nelementsflag == 0 || nwords != nelements+1)
        error->all(FLERR,"Incorrect SNAP parameter file");

      if (strcmp(keywd,"elems") == 0) {
        for (int ielem = 0; ielem < nelements; ielem++) {
          char* elemtmp = keyval;
          int n = strlen(elemtmp) + 1;
          elements[ielem] = new char[n];
          strcpy(elements[ielem],elemtmp);
          keyval = strtok(nullptr,"' \t\n\r\f");
        }
        elementsflag = 1;
      } else if (strcmp(keywd,"radelems") == 0) {
        for (int ielem = 0; ielem < nelements; ielem++) {
          radelem[ielem] = atof(keyval);
          keyval = strtok(nullptr,"' \t\n\r\f");
        }
        radelemflag = 1;
      } else if (strcmp(keywd,"welems") == 0) {
        for (int ielem = 0; ielem < nelements; ielem++) {
          wjelem[ielem] = atof(keyval);
          keyval = strtok(nullptr,"' \t\n\r\f");
        }
        wjelemflag = 1;
      }

    } else {

    // all other keywords take one value

      if (nwords != 2)
        error->all(FLERR,"Incorrect SNAP parameter file");

      if (strcmp(keywd,"nelems") == 0) {
        nelements = atoi(keyval);
        elements = new char*[nelements];
        memory->create(radelem,nelements,"mliap_snap_descriptor:radelem");
        memory->create(wjelem,nelements,"mliap_snap_descriptor:wjelem");
        nelementsflag = 1;
      } else if (strcmp(keywd,"rcutfac") == 0) {
        rcutfac = atof(keyval);
        rcutfacflag = 1;
//      } else if (strcmp(keywd,"twojmax") == 0) {
//        twojmax = atoi(keyval);
//        twojmaxflag = 1;
      } else if (strcmp(keywd,"nmax") == 0) {
          nmax = atoi(keyval);
          nmaxflag = 1;
      } else if (strcmp(keywd,"lmax") == 0) {
            lmax = atoi(keyval);
            lmaxflag = 1;
      } else if (strcmp(keywd,"alpha") == 0) {
          alpha = atof(keyval);
          alphaflag = 1;
      }
      else if (strcmp(keywd,"switchflag") == 0)
        switchflag = atoi(keyval);
      else if (strcmp(keywd,"bzeroflag") == 0)
        bzeroflag = atoi(keyval);
      else if (strcmp(keywd,"chemflag") == 0)
        chemflag = atoi(keyval);
      else if (strcmp(keywd,"bnormflag") == 0)
        bnormflag = atoi(keyval);
      else if (strcmp(keywd,"wselfallflag") == 0)
        wselfallflag = atoi(keyval);
      else
        error->all(FLERR,"Incorrect SO3 parameter file");

    }
  }

  if (!rcutfacflag || !nelementsflag ||
      !elementsflag || !radelemflag || !wjelemflag ||
	  !nmaxflag  || !lmaxflag || !alphaflag)
    error->all(FLERR,"Incorrect SO3 parameter file");

  // construct cutsq

  double cut;
  cutmax = 0.0;
  memory->create(cutsq,nelements,nelements,"mliap/descriptor/so3:cutsq");
  for (int ielem = 0; ielem < nelements; ielem++) {

    cut = 2.0*radelem[ielem]*rcutfac;
    cout << "rcutfac cut "<<rcutfac << " "<<cut << endl;
    if (cut > cutmax) cutmax = cut;
    cutsq[ielem][ielem] = cut*cut;
//    cutsq[ielem][ielem] = 9.0;
//    cout << "rcutfac " << rcutfac << " cutsq " << cut*cut << endl;
    for(int jelem = ielem+1; jelem < nelements; jelem++) {
      cut = (radelem[ielem]+radelem[jelem])*rcutfac;
      cutsq[ielem][jelem] = cutsq[jelem][ielem] = cut*cut;
//      cutsq[ielem][jelem] = cutsq[jelem][ielem] = 9.0;
    }

/*
	for(int jelem = ielem+1; jelem < nelements; jelem++) {
      cutsq[ielem][jelem] = cutsq[jelem][ielem] = rcutfac*rcutfac;
	}
*/
  }
  cout << "rcutfac*rcutfac "<< rcutfac*rcutfac << endl;
  cout << "cutsq[0][0] "<< cutsq[0][0] << endl;
  cout << "sizeof(wjelem) "<< sizeof(wjelem) << endl;
  cout << "wjelem[0] " << wjelem[0] << " wjelem[1] " << wjelem[1] << endl;
}

/* ----------------------------------------------------------------------
   compute descriptors for each atom
   ---------------------------------------------------------------------- */

void MLIAPDescriptorSO3::compute_descriptors(class MLIAPData* data)
{

   cout << "kkk in MLIAPDescriptorSO3::compute_descriptors" << endl;
   cout << "sizeof(wjelem) "<< sizeof(wjelem) << endl;
   cout << "wjelem[0] " << wjelem[0] << " wjelem[1] " << wjelem[1] << endl;
   cout << "chemflag " << chemflag << endl;
   cout << "natoms " << data->natoms << endl;
   cout << "numneighs 1 "<<data->numneighs[0] << " numneighs 1 "<<data->numneighs[1] << endl;
   cout << "nmax lmax " << nmax << " " << lmax << endl;
   cout << "rcutfac " << rcutfac << endl;
   cout << "alpha " << alpha << endl;

   int derivative=0,stress=0;
  // data->descriptors
   double *plist_r,*plist_i;
   plist_r=new double[(data->natoms)*(data->ndescriptors)];
   plist_i=new double[(data->natoms)*(data->ndescriptors)];

   int totali=(data->natoms)*(data->ndescriptors);
   for(int ti=0;ti<totali;ti++){
	   plist_r[ti]=0.0;
	   plist_i[ti]=0.0;
   }

   so3ptr->spectrum(data->natoms,data->numneighs,data->jelems,wjelem,
		    data->rij,
			nmax, lmax, rcutfac,alpha,derivative,stress,
			data->ndescriptors,plist_r,plist_i);

   for (int ii = 0; ii < data->natoms; ii++) {
     for (int icoeff = 0; icoeff < data->ndescriptors; icoeff++)
           data->descriptors[ii][icoeff] = plist_r[ii*(data->ndescriptors)+icoeff];
   }
   delete plist_r;
   delete plist_i;

// for print x
	cout.precision(5);
	cout <<std::scientific;
   cout << " data->descriptors " << endl;
   for (int ii = 0; ii < data->natoms; ii++) {
     for (int icoeff = 0; icoeff < data->ndescriptors; icoeff++){
		  cout << data->descriptors[ii][icoeff] << " ";
		  if((icoeff % 4)==3) cout << endl;
     }
     cout << endl;
   }
   cout << endl;
 // end for print x

/*// for dxdr test
   int derivative=1,stress=0;
  // data->descriptors
   double *plist_r,*plist_i,*dplist_r,*dplist_i;
   int numps = nmax*(nmax+1)*(lmax+1)/2;
   cout << "numps " << numps << endl;

   int npairs=0;
   for (int ii = 0; ii < data->natoms; ii++) {
       	npairs+=data->numneighs[ii];
   }
   cout << "npairs " << npairs << endl;



   plist_r=new double[(data->natoms)*(data->ndescriptors)];
   plist_i=new double[(data->natoms)*(data->ndescriptors)];
//   dplist_r=new double[(numps)*(data->ndescriptors)*3];
//   dplist_i=new double[(numps)*(data->ndescriptors)*3];
   dplist_r=new double[npairs*(data->ndescriptors)*3];
   dplist_i=new double[npairs*(data->ndescriptors)*3];

//   void SO3::spectrum_dxdr(int natoms,int* numneighs,int* jelems,double* wjelem,
//   		 double** rij,
//   		int nmax, int lmax,double rcut, double alpha,int npairs,
//   		int ncoefs,double* plist_r, double* plist_i,double* dplist_r, double* dplist_i)


   so3ptr->spectrum_dxdr(data->natoms,data->numneighs,data->jelems,wjelem,
		    data->rij,
			nmax, lmax, rcutfac,alpha,npairs,
			data->ndescriptors,plist_r,plist_i,dplist_r,dplist_i);

   delete dplist_r;
   delete dplist_i;
   delete plist_r;
   delete plist_i;
*/   // end for dxdr test
}

/* ----------------------------------------------------------------------
   compute forces for each atom
   ---------------------------------------------------------------------- */

void MLIAPDescriptorSO3::compute_forces(class MLIAPData* data)
{
	cout << "kkk in MLIAPDescriptorSO3::compute_forces " << endl;


    int derivative=1,stress=0;
  // data->descriptors
    double *plist_r,*plist_i,*dplist_r,*dplist_i;
    int numps = nmax*(nmax+1)*(lmax+1)/2;
    cout << "numps " << numps << endl;

    int npairs=0;
    for (int ii = 0; ii < data->natoms; ii++) {
		npairs+=data->numneighs[ii];
    }
    cout << "npairs " << npairs << endl;


    plist_r=new double[(data->natoms)*(data->ndescriptors)];
    plist_i=new double[(data->natoms)*(data->ndescriptors)];
//   dplist_r=new double[(numps)*(data->ndescriptors)*3];
//   dplist_i=new double[(numps)*(data->ndescriptors)*3];
    int totali=(data->natoms)*(data->ndescriptors);
    for(int ti=0;ti<totali;ti++){
 	   plist_r[ti]=0.0;
 	   plist_i[ti]=0.0;
    }
    dplist_r=new double[npairs*(data->ndescriptors)*3];
    dplist_i=new double[npairs*(data->ndescriptors)*3];
    totali=npairs*(data->ndescriptors)*3;
    for(int ti=0;ti<totali;ti++){
 	   dplist_r[ti]=0.0;
 	   dplist_i[ti]=0.0;
    }

//   void SO3::spectrum_dxdr(int natoms,int* numneighs,int* jelems,double* wjelem,
//   		 double** rij,
//   		int nmax, int lmax,double rcut, double alpha,int npairs,
//   		int ncoefs,double* plist_r, double* plist_i,double* dplist_r, double* dplist_i)


    so3ptr->spectrum_dxdr(data->natoms,data->numneighs,data->jelems,wjelem,
			data->rij,
			nmax, lmax, rcutfac,alpha,npairs,
			data->ndescriptors,plist_r,plist_i,dplist_r,dplist_i);

//    delete dplist_r;
//    delete dplist_i;
//    delete plist_r;
//    delete plist_i;


  double fij[3];
  double **f = atom->f;

  int ij = 0;
  double f0_0,f0_1,f0_2;
  double f1_0,f1_1,f1_2;
  f0_0=f0_1=f0_2=0.0;
  f1_0=f1_1=f1_2=0.0;
  for (int ii = 0; ii < data->natoms; ii++) {
     const int i = data->iatoms[ii];
     const int ielem = data->ielems[ii];

     // insure rij, inside, wj, and rcutij are of size jnum

     const int jnum = data->numneighs[ii];
/*
    cout << "sizeof(iatoms), sizeof(jatoms) "<< sizeof(data->iatoms) << " " <<sizeof(data->jatoms)<< endl;
    for(int ti=0;ti<sizeof(data->iatoms);ti++){
    	cout <<"ti,data->iatoms[ti] "<<ti<<" "<<data->iatoms[ti]<<endl;
    }
    for(int ti=0;ti<16;ti++){
    	cout <<"ti,data->jatoms[ti] "<<ti<<" "<<data->jatoms[ti]<<endl;
    }
    cout <<"ii i ielem jnum "<<ii<<" "<<i<<" "<<ielem<<" "<<jnum<< endl;
*/


     for (int jj = 0; jj < jnum; jj++) {
		  int j = data->jatoms[ij];

		  for(int ir=0;ir<3;ir++){
			fij[ir]=0.0;
			for(int icoeff=0;icoeff<data->ndescriptors;icoeff++){
			  fij[ir]+=data->betas[ii][icoeff]*dplist_r[(ij*(data->ndescriptors)+icoeff)*3+ir];
//	          fij[ir]+=0.0;
			}
		  }

		  cout <<"Force i j fij 0 1 2: "<< i << " " << j  << " " << fij[0]<< " " << fij[1] << " " << fij[2] << endl;
		  f[i][0] += fij[0];
		  f[i][1] += fij[1];
		  f[i][2] += fij[2];
		  f[j][0] -= fij[0];
		  f[j][1] -= fij[1];
		  f[j][2] -= fij[2];

		  if(i==0) {
			  f0_0 += fij[0];
			  f0_1 += fij[1];
			  f0_2 += fij[2];
		  }
		  if(i==1) {
			  f1_0 += fij[0];
			  f1_1 += fij[1];
			  f1_2 += fij[2];
		  }
		  if(j==0) {
			  f0_0 -= fij[0];
			  f0_1 -= fij[1];
			  f0_2 -= fij[2];
		  }
		  if(j==1) {
			  f1_0 -= fij[0];
			  f1_1 -= fij[1];
			  f1_2 -= fij[2];
		  }

		  // add in global and per-atom virial contributions
		  // this is optional and has no effect on force calculation

		  if (data->vflag)
			data->pairmliap->v_tally(i,j,fij,data->rij[ij]);
		  ij++;
     }

  }

  cout <<"F[0][0,1,2] "<< f[0][0] << " "<<f[0][1]<<" " << f[0][2] << endl;
  cout <<"F[1][0,1,2] "<< f[1][0] << " "<<f[1][1]<<" " << f[1][2] << endl;

  cout <<"f0_[0,1,2] "<< f0_0 << " "<<f0_1<<" " << f0_2<< endl;
  cout <<"f1_[0,1,2] "<< f1_0<< " "<<f1_1<<" " << f1_2 << endl;

}

/* ----------------------------------------------------------------------
   calculate gradients of forces w.r.t. parameters
   ---------------------------------------------------------------------- */

void MLIAPDescriptorSO3::compute_force_gradients(class MLIAPData* data)
{
	cout << "kkk in MLIAPDescriptorSO3::compute_force_gradients" << endl;
	/*
  int ij = 0;
  for (int ii = 0; ii < data->natoms; ii++) {
    const int i = data->iatoms[ii];
    const int ielem = data->ielems[ii];

    // insure rij, inside, wj, and rcutij are of size jnum

    const int jnum = data->numneighs[ii];
    snaptr->grow_rij(jnum);

    int ninside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      const int j = data->jatoms[ij];
      const int jelem = data->jelems[ij];

      const double *delr = data->rij[ij];

      snaptr->rij[ninside][0] = delr[0];
      snaptr->rij[ninside][1] = delr[1];
      snaptr->rij[ninside][2] = delr[2];
      snaptr->inside[ninside] = j;
      snaptr->wj[ninside] = wjelem[jelem];
      snaptr->rcutij[ninside] = sqrt(cutsq[ielem][jelem]);
      snaptr->element[ninside] = jelem; // element index for chem snap
      ninside++;
      ij++;
    }

    if (chemflag)
      snaptr->compute_ui(ninside, ielem);
    else
      snaptr->compute_ui(ninside, 0);

    snaptr->compute_zi();
    if (chemflag)
      snaptr->compute_bi(ielem);
    else
      snaptr->compute_bi(0);

    for (int jj = 0; jj < ninside; jj++) {
      const int j = snaptr->inside[jj];

      if(chemflag)
        snaptr->compute_duidrj(snaptr->rij[jj], snaptr->wj[jj],
                               snaptr->rcutij[jj],jj, snaptr->element[jj]);
      else
        snaptr->compute_duidrj(snaptr->rij[jj], snaptr->wj[jj],
                               snaptr->rcutij[jj],jj, 0);

      snaptr->compute_dbidrj();

      // Accumulate gamma_lk*dB_k/dRi, -gamma_lk**dB_k/dRj

      for (int inz = 0; inz < data->gamma_nnz; inz++) {
        const int l = data->gamma_row_index[ii][inz];
        const int k = data->gamma_col_index[ii][inz];
        data->gradforce[i][l]         += data->gamma[ii][inz]*snaptr->dblist[k][0];
        data->gradforce[i][l+data->yoffset] += data->gamma[ii][inz]*snaptr->dblist[k][1];
        data->gradforce[i][l+data->zoffset] += data->gamma[ii][inz]*snaptr->dblist[k][2];
        data->gradforce[j][l]         -= data->gamma[ii][inz]*snaptr->dblist[k][0];
        data->gradforce[j][l+data->yoffset] -= data->gamma[ii][inz]*snaptr->dblist[k][1];
        data->gradforce[j][l+data->zoffset] -= data->gamma[ii][inz]*snaptr->dblist[k][2];
      }

    }
  }

*/
}

/* ----------------------------------------------------------------------
   compute descriptor gradients for each neighbor atom
   ---------------------------------------------------------------------- */

void MLIAPDescriptorSO3::compute_descriptor_gradients(class MLIAPData* data)
{
	cout << "kkk in MLIAPDescriptorSO3::compute_descriptor_gradients" << endl;


}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void MLIAPDescriptorSO3::init()
{
  so3ptr->init();
}


/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double MLIAPDescriptorSO3::memory_usage()
{
  double bytes = 0;

  bytes += nelements*sizeof(double);            // radelem
  bytes += nelements*sizeof(double);            // welem
  bytes += nelements*nelements*sizeof(int);     // cutsq
//  bytes += snaptr->memory_usage();              // SNA object

  return bytes;
}

