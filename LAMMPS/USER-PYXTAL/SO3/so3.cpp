


#include <iostream>
#include <string>
#include <complex>
#include "so3.h"
#include <cmath>




#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "mkl.h"


using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;


SO3::SO3(LAMMPS*, double rcut,int vlmax,int vnmax) : Pointers(lmp)
{
    lmax=vlmax;
    nmax=vnmax;
	compute_ncoeff();
}

/* ---------------------------------------------------------------------- */

SO3::~SO3()
{

}
void SO3::compute_ncoeff()
{
//    ncoeff = 30;
    ncoeff = nmax*(nmax+1)*(lmax+1)/2;
}
void SO3::init()
{

//  init_clebsch_gordan();
  //   print_clebsch_gordan();
//  init_rootpqarray();
}
/*
void SO3::init_clebsch_gordan()
{
  double sum,dcg,sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  int idxcg_count = 0;
  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++)
      for(int j = j1 - j2; j <= MIN(twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; m1++) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2++) {

            // -c <= cc <= c

            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if(m < 0 || m > j) {
              cglist[idxcg_count] = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int z = MAX(0, MAX(-(j - j2 + aa2)
                                    / 2, -(j - j1 - bb2) / 2));
                 z <= MIN((j1 + j2 - j) / 2,
                          MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                 z++) {
              ifac = z % 2 ? -1 : 1;
              sum += ifac /
                (factorial(z) *
                 factorial((j1 + j2 - j) / 2 - z) *
                 factorial((j1 - aa2) / 2 - z) *
                 factorial((j2 + bb2) / 2 - z) *
                 factorial((j - j2 + aa2) / 2 + z) *
                 factorial((j - j1 - bb2) / 2 + z));
            }

            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = sqrt(factorial((j1 + aa2) / 2) *
                          factorial((j1 - aa2) / 2) *
                          factorial((j2 + bb2) / 2) *
                          factorial((j2 - bb2) / 2) *
                          factorial((j  + cc2) / 2) *
                          factorial((j  - cc2) / 2) *
                          (j + 1));

            cglist[idxcg_count] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
}
void SO3::init_rootpqarray()
{
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray[p][q] = sqrt(static_cast<double>(p)/q);
}
*/
void SO3::print_1di(int* arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}
void SO3::print_2d(string cstr,double* arr,int n){
	int i,j;

    cout << endl;
    cout << cstr << endl;
    for ( i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          cout << arr[n*i+j] << " " ;
    	}
    	cout << endl;
    }
}
void SO3::print_2d(string cstr,complex<double> *arr,int n, int n2){
	int i,j;

	cout.precision(5);
	cout <<std::scientific;
    cout << endl;
    cout << cstr << endl;
    for ( i=0;i<n;i++){
    	for ( j=0;j<n2;j++){
          cout << arr[n2*i+j] << " " ;
          if(j%2==1) cout << endl;
    	}
    	cout << endl;
    }
}
void SO3::print_2d(string cstr,double *arr,int n, int n2){
	int i,j;

	cout.precision(5);
	cout <<std::scientific;
    cout << endl;
    cout << cstr << endl;
    for ( i=0;i<n;i++){
    	for ( j=0;j<n2;j++){
          cout << arr[n2*i+j] << " " ;
          if(j%4==3) cout << endl;
    	}
    	cout << endl;
    }
}
void SO3::print_3d(string cstr,double* arr,int m,int n,int d,int cut){

	int i,j,k;

	cout.precision(5);
	cout <<std::scientific;
    cout << endl;
    cout << cstr << endl;

	for(i=0;i<m;i++){
		if(cut !=0 && i> cut) break;
		if(i==0) cout << "[[";
		else cout << " [";
		for(j=0;j<n;j++){
		   if(j==0) cout << "[";
		   else cout << "  [";
		   for(k=0;k<d;k++){
		      cout << arr[(i*n+j)*d+k] << " ";
		   }
		   if(j==n-1) cout << "]";
		   else cout << "]" << endl;
		}
		if(i==m-1) cout << "]]" << endl;
		else cout << "]" << endl;
		cout << endl;
	}
}
void SO3::print_1c(complex<double> *arr,int m){
	int i;

	cout << "[";
	for(i=0;i<m;i++){
		cout << arr[i] << " ";
	}
	cout << "]" << endl;
}
void SO3::swap(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}
int SO3::partition (double arr[], int low, int high, double arrv[],int n)
{
    double pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
    int vi;
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] >= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
            for(vi=0;vi<n;vi++){
            	swap(&arrv[vi*n+i], &arrv[vi*n+j]);
            }
        }
    }
    swap(&arr[i + 1], &arr[high]);
    for(vi=0;vi<n;vi++){
    	swap(&arrv[vi*n+i+1], &arrv[vi*n+high]);
    }
    return (i + 1);
}
void SO3::quickSort(double arr[], int low, int high, double arrv[], int n)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high, arrv,n);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1, arrv,n);
        quickSort(arr, pi + 1, high, arrv,n);
    }
}
void SO3::W(int nmax, double *arr){

	int alpha,beta,temp1,temp2;

	for(alpha=1;alpha<nmax+1;alpha++){
		temp1=(2*alpha+5)*(2*alpha+6)*(2*alpha+7);
		for(beta=1;beta<alpha+1;beta++){
            temp2 = (2*beta+5)*(2*beta+6)*(2*beta+7);
            arr[(alpha-1)*nmax+beta-1] = sqrt(temp1*temp2)/(5+alpha+beta)/(6+alpha+beta)/(7+alpha+beta);
            arr[(beta-1)*nmax+alpha-1] = arr[(alpha-1)*nmax+beta-1]	;
		}
	}
//	print_2d("arr",arr,nmax);

    char Nchar='V';
    char charN='N';
    char charT='T';
    int i,j,k,l,totaln,n=nmax;
    double *outeig= new double[n];
    double *outeigvec= new double[n*n];

    double *sqrtD=new double[n*n];
    double *tempM=new double[n*n];

    int *IPIV = new int[n];
    double *eigReal=new double[n];
    double *eigImag=new double[n];

    int lwork=6*n;
    double *vl=new double[n*n];
    double *vr=new double[n*n];
    double *work=new double[lwork];
    int info;

    //cout << "after inverse" << endl;
    dgetrf(&n,&n,arr,&n,IPIV,&info);
    dgetri(&n,arr,&n,IPIV,work,&lwork,&info);


    // calculate eigenvalues using the DGEEV subroutine
    dgeev(&Nchar,&Nchar,&n,arr,&n,outeig,eigImag,
          vl,&n,vr,&n,
          work,&lwork,&info);
    // check for errors
    if (info!=0){
      cout << "Error: dgeev returned error code " << info << endl;
      return ;
    }

    for( i=0;i<n;i++){
       for( j=0;j<n;j++){
    	   outeigvec[i*n+j]=vl[j*n+i];
       }
    }
    // output eigenvalues to stdout

    quickSort(outeig,0,n-1,outeigvec,n);
//    cout << "--- Eigenvalues ---" << endl;
//    for ( i=0;i<n;i++){
//      cout << outeig[i] << " ";
//    }
//    print_2d("eigen vectors",outeigvec,n);

    for ( i=0;i<n;i++){
    	for ( j=0;j<n;j++){
          if(i==j) sqrtD[i*n+j]=sqrt(outeig[i]);
          else sqrtD[i*n+j]=0.0;
    	}
    }
//    print_2d("sqrtD",sqrtD,n);


// ** to avoid row major issue, implemented direct matrix dot.
//    double dpone = 1.0, dmone = 0.0;
//    dgemm_ (&charN, &charT, &n, &n, &n, &dpone, outeigvec, &n, sqrtD, &n, &dmone,
//    tempM, &n);
    double dtemp;
    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=outeigvec[i*n+k]*sqrtD[k*n+j];

	        tempM[i*n+j]=dtemp;
    	}
    }

//    print_2d("V*sqrtD",tempM,n);

    dgetrf_(&n,&n,outeigvec,&n,IPIV,&info);
    dgetri_(&n,outeigvec,&n,IPIV,work,&lwork,&info);

//    print_2d("inv V",outeigvec,n);

    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		dtemp=0;
    		for(k=0;k<n;k++) dtemp+=tempM[i*n+k]*outeigvec[k*n+j];

	        arr[i*n+j]=dtemp;
    	}
    }
//    print_2d("out data",arr,n);

    delete outeig;
    delete outeigvec;

    delete sqrtD;
    delete tempM;

    delete IPIV;
    delete eigReal;
    delete eigImag;

    delete vl;
    delete vr;
    delete work;

}
void SO3::compute_pi(int nmax,int lmax, complex<double> *clisttot, int lcl1,int lcl2,
		complex<double> *plist, int lpl1,int lpl2,int indpl){

	//cout << "pi "<< M_PI << endl;
	int n1,n2,j,l,m,i=0;
	double norm;
	for(n1=0;n1<nmax;n1++){
		for(n2=0;n2<n1+1;n2++){
			j=0;
			for(l=0;l<lmax+1;l++){
				 norm = 2.0*sqrt(2.0)*M_PI/sqrt(2.0*l+1.0);
				 //cout<<"norm l "<<norm << " "<<l<<endl;

			     for(m=-l;m<l+1;m++){
				      //cout << "n1 n2 m "<<n1<<" "<<n2<<" "<<m<<endl;
				      //cout << "clist conj(clist) "<< clisttot[lcl2*n2+j] << " "<<conj(clisttot[lcl2*n2+j])<<endl;
				      plist[lpl2*indpl+i] += clisttot[lcl2*n1+j] * conj(clisttot[lcl2*n2+j])*norm;
			          j += 1;
			     }
			     i += 1;
			}
		}
	}
}
double SO3::phi(double r,int alpha,double rcut){

	return pow((rcut-r),(alpha+2))/sqrt(2*pow(rcut,(2*alpha+7))/(2*alpha+5)/(2*alpha+6)/(2*alpha+7));
}
complex<double> SO3::g(double r,int n,int nmax,double rcut,double *w,int lw1, int lw2){

	complex<double> Sum;
	Sum={0.0,0.0};
	int alpha;
	for(alpha=1;alpha<nmax+1;alpha++){
		Sum += w[(n-1)*lw1+alpha-1]*phi(r, alpha, rcut);
	}
	return Sum;
}
double SO3::modifiedSphericalBessel1(double r, int n, int derivative){

	double *temp_arr;
	double dval;
	int i;

	if(derivative==0){
		if(n==0){
			return sinh(r)/r;
		}else if(n==1){
			return (r*cosh(r)-sinh(r))/(r*r);
		}else{
			temp_arr=new double[n+1];
			for(i=0;i<n+1;i++) temp_arr[i]=0;
			temp_arr[0] = sinh(r)/r;
			temp_arr[1] = (r*cosh(r)-sinh(r))/(r*r);
			for(i=2;i<n+1;i++){
				temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1];
			}
			dval=temp_arr[n];
			delete temp_arr;
			return dval;
		}
	}else{
		if(n==0){
			return (r*cosh(r)-sinh(r))/(r*r);
		}else{
			temp_arr=new double[n+2];
			for(i=0;i<n+2;i++) temp_arr[i]=0;
			temp_arr[0] = sinh(r)/r;
			temp_arr[1] = (r*cosh(r)-sinh(r))/(r*r);
			for(i=2;i<n+2;i++){
				temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1];
			}
			dval=(n*temp_arr[n-1] + (n+1)*temp_arr[n+1]) / (2*n+1);
			delete temp_arr;
			return dval;
		}

	}
}

complex<double> SO3::integrand(double r, double ri, double alpha, double rcut,
		int n, int l, int nmax, double *w,int lw1,int lw2, int derivative){

    if (derivative == 0)
        return r*r*g(r, n, nmax, rcut, w,lw1,lw2)*exp(-alpha*r*r)*modifiedSphericalBessel1(2*alpha*r*ri, l, 0);
    else
        return r*r*r*g(r, n, nmax, rcut, w,lw1,lw2)*exp(-alpha*r*r)*modifiedSphericalBessel1(2*alpha*r*ri, l, 1);
}

complex<double>  SO3::get_radial_inner_product(double ri, double alpha, double rcut,
		int n, int l, int nmax, double *w,int lw1,int lw2,int derivative){
	double x,xi;
	complex<double> integral={0.0,0.0};
	int i,BigN;
	BigN=(n+l+1)*10;
	for(i=1;i<BigN+1;i++){
		x = cos((2*i-1)*M_PI/2/BigN);
		xi = rcut/2*(x+1);
		integral += sqrt(1-x*x)*integrand(xi, ri, alpha, rcut, n, l, nmax, w,lw1,lw2, derivative);
	}
	integral *= rcut/2*M_PI/BigN;
	return integral;
}
int SO3::_Wigner_index(int twoj, int twomp, int twom){
    return int(twoj*((2*twoj + 3) * twoj + 1) / 6) + int((twoj + twomp)/2) * (twoj + 1) + int((twoj + twom) /2);
}

double SO3::_Wigner_coefficient(int twoj, int twomp, int twom){
	//cout <<"_Wigner_coefficient twoj twomp twom "<< twoj <<" " << twomp << " " <<twom << endl;
	//cout <<"_coeff " << Wigner_data[_Wigner_index(twoj, twomp, twom)] << endl;
    return Wigner_data[_Wigner_index(twoj, twomp, twom)];
}

double SO3::Wigner_coefficient(int j, int mp, int m){
    return _Wigner_coefficient(int(round(2*j)), int(round(2*mp)), int(round(2*m)));

}

complex<double> SO3::Wigner_D(complex<double> vRa, complex<double> vRb, int twol, int twomp, int twom){

	double ra, phia, rb,phib;
	ra=abs(vRa);
	phia=arg(vRa);

	rb=abs(vRb);
	phib=arg(vRb);

	double epsilon = pow(10,-15);
	complex<double> czero={0.0,0.0};

//    cout <<"Wigner_D"<<endl;
//    cout <<vRa<<" "<<vRb<<" "<<twol<<" "<<twomp<<" "<<twom<<" "<<ra
//    		<<" "<<phia<<" "<<rb<<" "<<phib<<" "<<epsilon<<endl;

    if(ra <= epsilon){
        if(twomp != -twom || abs(twomp) > twol || abs(twom) > twol){
            return czero;
        }else{
            if((twol - twom) % 4 == 0){
                return pow(vRb,twom);
            }else{
                return -pow(vRb,twom);
            }
        }
    }else if(rb <= epsilon){
        if(twomp != twom || abs(twomp) > twol || abs(twom) > twol){
            return czero;
        }else{
            return pow(vRa,twom);
        }
    }else if(ra < rb){
    	double x = - ra*ra / rb / rb;
    	if (abs(twomp) > twol || abs(twom) > twol){
    	    return czero;
    	}else{
    		complex<double> Prefactor = polar(
    				 _Wigner_coefficient(twol, -twomp, twom)
    		                * pow(rb,(twol - (twom+twomp)/2))
    		                * pow(ra, ((twom+twomp)/2)),
    		                phib * (twom - twomp)/2 + phia * (twom + twomp)/2);
    		if(Prefactor==czero){
    			return czero;
    		}else{
    			int  l = twol/2;
    			int  mp = twomp/2;
    			int  m = twom/2;
    			int kmax = round(min(l-mp, l-m));
    			int kmin = round(max(0, -mp-m));
    			if ((twol - twom) %4 != 0){
    			     Prefactor *= -1;
    			}
    			double Sum = 1/fac_arr[int(round(kmax))]/fac_arr[int(round(l-m-kmax))]
				              /fac_arr[int(round(mp+m+kmax))]/fac_arr[int(round(l-mp-kmax))];
    			for(int k=kmax-1;k>kmin-1;k--){
    				Sum *=x;
    				Sum += 1/fac_arr[int(round(k))]/fac_arr[int(round(l-m-k))]
							/fac_arr[int(round(mp+m+k))]/fac_arr[int(round(l-mp-k))];
    			}
    			Sum *= pow(x,kmin);
    			return Prefactor * Sum;
    		}
    	}
    }else{
    	//cout <<"ra>rb"<<endl;
    	double x = - rb*rb / (ra * ra);
    	//cout <<"x"<<x<<endl;
    	//cout<<abs(twomp)<<" "<<twol<<" "<<abs(twom)<<" "<<twol<<endl;
    	if (abs(twomp) > twol || abs(twom) > twol){
    	    return czero;
    	}else{
    		complex<double> Prefactor = polar(
    				 _Wigner_coefficient(twol, twomp, twom)
    		                * pow(ra,(twol - twom/2 + twomp/2))
    		                * pow(rb, (twom/2 - twomp/2)),
							phia * (twom + twomp)/2 + phib * (twom - twomp)/2);
    		//cout<<"Prefactor "<<Prefactor<<endl;
    		if(Prefactor==czero){
    			return czero;
    		}else{
    			int  l = twol/2;
    			int  mp = twomp/2;
    			int  m = twom/2;
    			int kmax = round(min(l + mp, l - m));
    			int kmin = round(max(0, mp-m));
    			//cout<<l<<" "<<mp<<" "<<m<<" "<<kmax<<" "<<kmin<<endl;
    			double Sum = 1/fac_arr[int(round(kmax))]/fac_arr[int(round(l+mp-kmax))]
				              /fac_arr[int(round(l-m-kmax))]/fac_arr[int(round(-mp+m+kmax))];
    			//cout <<"Sum 1 "<<Sum << endl;
    			for(int k=kmax-1;k>kmin-1;k--){
    				Sum *=x;
    				//cout << "k Sum_1 "<< Sum << endl;
    				Sum += 1/fac_arr[int(round(k))]/fac_arr[int(round(l+mp-k))]
							/fac_arr[int(round(l-m-k))]/fac_arr[int(round(-mp+m+k))];
    				//cout << "k Sum_2 "<< Sum << endl;
    			}
    			//cout <<"Sum 2 "<<Sum<<endl;
    			Sum *= pow(x,kmin);
    			//cout <<"Sum"<<" "<<Sum<<endl;
    			return Prefactor * Sum;
    		}
    	}
    }

}

complex<double> SO3::sph_harm(complex<double> Ra, complex<double> Rb, int l, int m){
	//cout <<"sph_harm" << endl;
	//cout <<Ra<<" "<<Rb <<" "<<l << " "<<m<<endl;

	complex<double> ctemp;
//	ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
//	cout << "ctemp "<<ctemp<<endl;
//	ctemp=conj(ctemp);
//	cout << "conj(ctemp) "<<ctemp<<endl;
	if(m%2==0){
		ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
//		cout <<"Wigner_D return "<<ctemp<<endl;
		ctemp=conj(ctemp);
//		cout <<"mid "<<sqrt((2.0*l+1.0)/4.0/M_PI)<<endl;
//		cout <<"sph_harm return "<< ctemp * sqrt((2*l+1)/4/M_PI) * pow(-1.0,m)<<endl;
		return ctemp * sqrt((2.0*l+1.0)/4.0/M_PI) * pow(-1.0,m);
	}else{
		ctemp=Wigner_D(Ra, Rb, 2*l, 0, -2*m);
		ctemp=conj(ctemp);
		return ctemp * sqrt((2.0*l+1.0)/4.0/M_PI) * pow(-1.0,m+1);
	}
}
//compute_dpidrj(nmax, lmax, clisttot,nmax,numYlms, dclist,nmax,numYlms,3,
//                                     tempdp,numps,3);
void SO3::compute_dpidrj(int nmax, int lmax, complex<double> *clisttot,int lctot1,int lctot2,
		complex<double> *dclist,int ldcli1,int ldcli2,int ldcli3,
		complex<double> *dplist,int dpli1,int dpli2){

	complex<double> temp;
	double norm;
	int i,n1,n2,j,l,m,ii;
	i=0;
	for(n1=0;n1<nmax;n1++){
		for(n2=0;n2<n1+1;n2++){
			j=0;
			for(l=0;l<lmax+1;l++){
				norm = 2.0*sqrt(2.0)*M_PI/sqrt(2.0*l+1.0);
				for(m=-l;m<l+1;m++){
					for(ii=0;ii<3;ii++){
                     temp = dclist[(n1*ldcli2+j)*ldcli3+ii] * conj(clisttot[n2*lctot2+j]);
                     temp += clisttot[n1*lctot2+j] * conj(dclist[(n2*ldcli2+j)*ldcli3+ii]);
                     temp *= norm;
                     dplist[i*dpli2+ii] += temp;
					}
					j+=1;
				}
				i += 1;
			}
		}
	}

}
//void SO3::compute_carray_wD(x, y, z, r, alpha, rcut, nmax, lmax, w,nmax,nmax,
//										clist,nmax,numYlms,dclist,nmax,numYlms,3)
int SO3::get_sum(int istart,int iend, int id,int imult){

	int ires=0;
	int i;
	for(i=istart;i<iend;i=i+id){
		ires += i*imult;
	}
	return ires;

}
void SO3::compute_carray_wD(double x, double y, double z, double ri, double alpha, double rcut,
		int nmax, int lmax, double *w, int lw1, int lw2,
		complex<double> *clist,int lcli1,int lcli2,
		complex<double> *dclist,int ldcli1,int ldcli2,int ldcli3){

	double rvec[3];
	rvec[0]=x;rvec[1]=y;rvec[2]=z;
	double dexpfac[3];

	complex<double> dr_int[3];


	double theta,phi,atheta,btheta,expfac;
	complex<double> aphi,Ra,Rb,r_int,Ylm,r_int_temp;
	complex<double> *Ylms,*dYlm;
	int ellpl1,ellm1;

	int n,i,l,m,totali;
	theta=acos(z/ri);
	phi=atan2(y,x);

	atheta = cos(theta/2);
	btheta = sin(theta/2);

	aphi = {cos(phi/2), sin(phi/2)};
    Ra = atheta*aphi;
    Rb = btheta*aphi;

    totali=(lmax+2)*(lmax+2);
    Ylms=new complex<double>[totali];
    for(int tn=0;tn<totali;tn++) Ylms[tn]={0.0,0.0};

	i=0;
	for(l=0;l<lmax+2;l++){
		for(m=-l;m<l+1;m++){
			 //cout <<Ra << " "<<Rb<<" "<<l<<" "<<m<<endl;
			 Ylms[i] = sph_harm(Ra, Rb, l, m);
			 //cout << "Ylm" << " "<<Ylm<<endl;
			 i+=1;
		}
	}

    totali=(lmax+1)*(lmax+1)*3;
    dYlm=new complex<double>[totali];
    for(int tn=0;tn<totali;tn++) dYlm[tn]={0.0,0.0};

    complex<double> xcov0,xcovpl1,xcovm1;
    complex<double> comj={0.0,1.0};
    i=1;
    for(l=1;l<lmax+1;l++){
       ellpl1=get_sum(0,l+2,1,2);
       ellm1=get_sum(0,l,1,2);
//       cout << "ellpl1 ellm1 l "<< ellpl1 << " " << ellm1 << " "<< l << endl;
       for(m=-l;m<l+1;m++){
           xcov0 = -sqrt(((l+1.0)*(l+1.0)-m*m)/(2.0*l+1.0)/(2.0*l+3.0))*l*Ylms[ellpl1+m]/ri;
           if (abs(m) <= l-1.0)
               xcov0 += sqrt((l*l-m*m)/(2.0*l-1.0)/(2.0*l+1.0))*(l+1.0)*Ylms[ellm1+m]/ri;

           xcovpl1 = -sqrt((l+m+1.0)*(l+m+2.0)/2.0/(2.0*l+1.0)/(2.0*l+3.0))*l*Ylms[ellpl1+m+1]/ri;
           if (abs(m+1) <= l-1.0)
               xcovpl1 -= sqrt((l-m-1.0)*(l-m)/2.0/(2.0*l-1.0)/(2.0*l+1.0))*(l+1.0)*Ylms[ellm1+m+1]/ri;

           xcovm1 = -sqrt((l-m+1.0)*(l-m+2.0)/2.0/(2.0*l+1.0)/(2.0*l+3.0))*l*Ylms[ellpl1+m-1]/ri;
           if (abs(m-1.0) <= l-1.0)
               xcovm1 -= sqrt((l+m-1.0)*(l+m)/2.0/(2.0*l-1.0)/(2.0*l+1.0))*(l+1.0)*Ylms[ellm1+m-1]/ri;

           dYlm[i*3+0] = 1.0/sqrt(2.0)*(xcovm1-xcovpl1);
           dYlm[i*3+1] = comj/sqrt(2.0)*(xcovm1+xcovpl1);
           dYlm[i*3+2] = xcov0;
//           cout << "dYlm 0 1 2 i "<<dYlm[i*3+0]<< " "<<dYlm[i*3+1]<< " "<<dYlm[i*3+2]<< " "<<i << endl;

           i += 1;
       }
    }

    expfac = 4*M_PI*exp(-alpha*ri*ri);
    for(int ii=0;ii<3;ii++){
    	dexpfac[ii]=-2.0*alpha*expfac*rvec[ii];
    }

    //cout << "compute_carray" << endl;
    //cout <<"x y z "<<x<<" "<<y<<" "<<z<<endl;
    //cout << theta<<" "<<phi<<" "<<atheta<<" "<<btheta<<" "
    //		<<aphi<<" "<<Ra<<" "<<Rb<<" "<<expfac<<endl;
    for(n=1;n<nmax+1;n++){
    	i=0;
    	for(l=0;l<lmax+1;l++){
    		r_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w,lw1,lw2, 0);
    		r_int_temp = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w,lw1,lw2, 1);
//    		cout <<"r_int_temp n l "<< r_int_temp << " " << n << " " << l << endl;
    		for(int ii=0;ii<3;ii++){
    		  dr_int[ii] = r_int_temp*2.0*alpha*rvec[ii]/ri;
    		}
    		//cout <<"r_int"<< " "<<r_int<<endl;
    		for(m=-l;m<l+1;m++){
    			 //cout <<Ra << " "<<Rb<<" "<<l<<" "<<m<<endl;

    			 clist[(n-1)*lcli2+i] += r_int*Ylms[i]*expfac;
    			 dclist[((n-1)*ldcli2+i)*ldcli3+0] +=r_int*Ylms[i]*dexpfac[0] + dr_int[0]*Ylms[i]*expfac + r_int*expfac*dYlm[i*3+0];
    			 dclist[((n-1)*ldcli2+i)*ldcli3+1] +=r_int*Ylms[i]*dexpfac[1] + dr_int[1]*Ylms[i]*expfac + r_int*expfac*dYlm[i*3+1];
    			 dclist[((n-1)*ldcli2+i)*ldcli3+2] +=r_int*Ylms[i]*dexpfac[2] + dr_int[2]*Ylms[i]*expfac + r_int*expfac*dYlm[i*3+2];
//    			 if(n==1 && i==1) return;
    			 i+=1;
    		}
    	}
    }
	delete Ylms;
	delete dYlm;

}

void SO3::compute_carray(double x, double y, double z, double ri, double alpha, double rcut,
		int nmax, int lmax, double *w, int lw1, int lw2,
		complex<double> *clist,int lcli1,int lcli2){

	double theta,phi,atheta,btheta,expfac;
	complex<double> aphi,Ra,Rb,r_int,Ylm;
	int n,i,l,m;
	theta=acos(z/ri);
	phi=atan2(y,x);

	atheta = cos(theta/2);
	btheta = sin(theta/2);

	aphi = {cos(phi/2), sin(phi/2)};
    Ra = atheta*aphi;
    Rb = btheta*aphi;

    expfac = 4*M_PI*exp(-alpha*ri*ri);

//    cout << "compute_carray" << endl;
//    cout <<"x y z "<<x<<" "<<y<<" "<<z<<endl;
//    cout << theta<<" "<<phi<<" "<<atheta<<" "<<btheta<<" "
//    		<<aphi<<" "<<Ra<<" "<<Rb<<" "<<expfac<<endl;
    for(n=1;n<nmax+1;n++){
    	i=0;
    	for(l=0;l<lmax+1;l++){
    		r_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w,lw1,lw2, 0);
//    		cout <<"r_int"<< " "<<r_int<<endl;
    		for(m=-l;m<l+1;m++){
//    			 cout <<Ra << " "<<Rb<<" "<<l<<" "<<m<<endl;
    			 Ylm = sph_harm(Ra, Rb, l, m);
//    			 cout << "Ylm" << " "<<Ylm<<endl;
    			 clist[(n-1)*lcli2+i] += r_int*Ylm*expfac;
//    			 if(n==1 && i==1) return;
    			 i+=1;
    		}
    	}
    }
}
int SO3::read_Wigner_data(){

        FILE * filp = fopen("Wigner_coefficients.dat", "rb");
        if(filp==NULL){
                cout << "file open error" << endl;
                fclose(filp);
                return 0;
        }
        int sizei=sizeof(int);
        int sized=sizeof(double);
        int total;
        int bytes_read = fread(&total, sizeof(int), 1, filp);
        if(bytes_read <=0 ){
                cout << "file read error" << endl;
                fclose(filp);
                return 0;
        }
        Wigner_data = new double[total];
        int i;
        for(i=0;i<total;i++){
          fread(&Wigner_data[i], sized, 1, filp);
//        cout << i << " " << Wigner_data[i] << endl;
        }
        fclose(filp);
//        cout << endl;
//      cout <<  wdata[_Wigner_index(2,1,0)] << endl;
 //       delete wdata;
        return 0;
}
void SO3::spectrum(int natoms,int* numneighs,int* jelems,double* wjelem,
		 double** rij,
		int nmax, int lmax,double rcut, double alpha,int derivative,int stress,
		int ncoefs,double* plist_r, double* plist_i){


	complex<double> *plist,*dplist,*pstress,*clisttot,*clist,*dclist;
	plist=new complex<double>[natoms*ncoefs];
	double *w;

	cout << "kkk in SO3::spectrum " << endl;
	cout << "ncoefs " << ncoefs << endl;

//	cout << "rij[0][1,2,3] " << rij[0][0] << " " << rij[0][1] << " " << rij[0][2] << endl;
//	cout << "rij[7][1,2,3] " << rij[7][0] << " " << rij[7][1] << " " << rij[7][2] << endl;
	read_Wigner_data();

	int i,j,k,l,ti;
	int totali;
	totali=natoms*ncoefs;
    for(i=0;i<totali;i++){
         plist[i]={plist_r[i],plist_i[i]};
    }

//    print_1c(plist,lpli1*lpli2);
//    int npairs=lne1;
//    int nneighbors=lne2;
    int numYlms=(lmax+1)*(lmax+1);
    cout << "numYlms " << numYlms << endl;
    clisttot=new complex<double>[nmax*numYlms];
    clist=new complex<double>[nmax*numYlms];
//    dclist=new complex<double>[nneighbors*nmax*numYlms*3];
    totali=nmax*numYlms;
    for(i=0;i<totali;i++){
    	clisttot[i]={0.0,0.0};
    	clist[i]={0.0,0.0};
    }
//    totali=nneighbors*nmax*numYlms;
//    for(i=0;i<totali;i++){
//    	dclist[i]={0.0,0.0};
//    }
    w=new double[nmax*nmax];
    totali=nmax*nmax;
    for(i=0;i<totali;i++) w[i]=0.0;

    W(nmax,w);
    //print_2d("w",w,nmax);
    //print_2d("o before clisttot",clisttot,nmax,numYlms);



    int numps,nstart,nsite,n,weight,neighbor;
    complex<double> *tempdp;
    double isite;
    double x,y,z,r;

    if(derivative==1){
		cout << "Error: not implemented" << endl;
		return;
    	if(stress==1){

    	}else{

    	}


    }else{
    	  int ipair = 0;
    	  for (int ii = 0; ii < natoms; ii++) {
    		  totali=nmax*numYlms;
    		  for(int tn=0;tn<totali;tn++){
    		     clisttot[tn]={0.0,0.0};
    		  }
//    	    const int ielem = data->ielems[ii];
    		  const int jelem = jelems[ipair];
    		  weight=wjelem[jelem];
    		  cout << "ipair jelem weight "<<ipair << " "<< jelem <<" "<<weight<<endl;
              cout << "ii, numneights[ii] "<< ii << " "<< numneighs[ii]<< endl;
    		  for(neighbor=0;neighbor<numneighs[ii];neighbor++){

            	  x = rij[ipair][0];
            	  y = rij[ipair][1];
            	  z = rij[ipair][2];
            	  ipair++;

                  r = sqrt(x*x + y*y + z*z);
//                  cout <<"x,y,z,r "<<x<<" "<<y<<" "<<z<<" "<<r<<endl;

                  if(r<pow(10,-8)) continue;
                  totali=nmax*numYlms;
                  for(ti=0;ti<totali;ti++) clist[ti]={0.0,0.0};

                  compute_carray(x, y, z, r, alpha, rcut, nmax, lmax, w,nmax,nmax,
                                                        clist,nmax,numYlms);

                  totali=nmax*numYlms;
                  for(int tn=0;tn<totali;tn++){
                	  clist[tn] = clist[tn]*double(weight);
                  }
//                  print_2d("clist",clist,nmax,numYlms);

                  for(int tn=0;tn<totali;tn++){
                	  clisttot[tn] += clist[tn];
                  }

              }

              compute_pi(nmax,lmax,clisttot,nmax,numYlms,plist,natoms,ncoefs,ii);
//              cout << "ii " << ii << endl;
//              print_2d("clisttot",clisttot,nmax,numYlms);
    	  }


		  totali=natoms*ncoefs;
	      for(i=0;i<totali;i++){
	         plist_r[i]=real(plist[i]);
	         plist_i[i]=imag(plist[i]);
	      }

//	      print_2d(" in so3 plist_r",plist_r,natoms,ncoefs);

		  return;

    }

}
void SO3::spectrum_dxdr(int natoms,int* numneighs,int* jelems,double* wjelem,
		 double** rij,
		int nmax, int lmax,double rcut, double alpha,int npairs,
		int ncoefs,double* plist_r, double* plist_i,double* dplist_r, double* dplist_i){

	int i,j,k,l,ti;
	int totali;
	complex<double> *plist,*dplist,*pstress,*clisttot,*clist,*dclist;
	plist=new complex<double>[natoms*ncoefs];
	dplist=new complex<double>[npairs*ncoefs*3];
	double *w;
	totali=npairs*ncoefs*3;
    for(i=0;i<totali;i++){
         dplist[i]={0.0,0.0};
    }

	cout << "kkk in SO3::spectrum_dxdr " << endl;
	cout << "ncoefs " << ncoefs << endl;

//	cout << "rij[0][1,2,3] " << rij[0][0] << " " << rij[0][1] << " " << rij[0][2] << endl;
//	cout << "rij[7][1,2,3] " << rij[7][0] << " " << rij[7][1] << " " << rij[7][2] << endl;
	read_Wigner_data();


	totali=natoms*ncoefs;
    for(i=0;i<totali;i++){
         plist[i]={plist_r[i],plist_i[i]};
    }
/*    int npairs=0;
    for (int ii = 0; ii < natoms; ii++) {
    	npairs+=numneighs[ii];
    }
    cout << "npairs " << npairs << endl;
*/
//    print_1c(plist,lpli1*lpli2);
//    int npairs=lne1;
//    int nneighbors=lne2;
    int numYlms=(lmax+1)*(lmax+1);
    cout << "numYlms " << numYlms << endl;
    clisttot=new complex<double>[nmax*numYlms];
    clist=new complex<double>[nmax*numYlms];
    dclist=new complex<double>[nmax*numYlms*3];
    totali=nmax*numYlms;
    for(i=0;i<totali;i++){
    	clisttot[i]={0.0,0.0};
    	clist[i]={0.0,0.0};
    }
    totali=nmax*numYlms*3;
    for(i=0;i<totali;i++){
    	dclist[i]={0.0,0.0};
    }
    w=new double[nmax*nmax];
    totali=nmax*nmax;
    for(i=0;i<totali;i++) w[i]=0.0;

    W(nmax,w);
    //print_2d("w",w,nmax);
    //print_2d("o before clisttot",clisttot,nmax,numYlms);



    int numps,nstart,nsite,n,weight,neighbor;
    complex<double> *tempdp;

    numps=nmax*(nmax+1)*(lmax+1)/2;
    cout << "numps " << numps << endl;
    tempdp=new complex<double>[numps*3];
    double isite;
    double x,y,z,r;


	int ipair = 0;
	int idpair =0;
	for (int ii = 0; ii < natoms; ii++) {
	  totali=nmax*numYlms;
	  for(int tn=0;tn<totali;tn++){
		 clisttot[tn]={0.0,0.0};
	  }
//    	    const int ielem = data->ielems[ii];
	  const int jelem = jelems[ipair];
	  weight=wjelem[jelem];
	  cout << "ipair jelem weight "<<ipair << " "<< jelem <<" "<<weight<<endl;
	  for(neighbor=0;neighbor<numneighs[ii];neighbor++){

		  x = rij[ipair][0];
		  y = rij[ipair][1];
		  z = rij[ipair][2];
		  ipair++;


		  r = sqrt(x*x + y*y + z*z);
		  if(r<pow(10,-8)) continue;
		  totali=nmax*numYlms;
		  for(ti=0;ti<totali;ti++) clist[ti]={0.0,0.0};

		  compute_carray(x, y, z, r, alpha, rcut, nmax, lmax, w,nmax,nmax,
												clist,nmax,numYlms);

		  totali=nmax*numYlms;
		  for(int tn=0;tn<totali;tn++){
			  clist[tn] = clist[tn]*double(weight);
		  }

		  for(int tn=0;tn<totali;tn++){
			  clisttot[tn] += clist[tn];
		  }

	  }
	  compute_pi(nmax,lmax,clisttot,nmax,numYlms,plist,natoms,ncoefs,ii);

/*
	  cout <<"plist[ii] "<<ii << endl;
	  for(int kk=0;kk<ncoefs;kk++){
		  cout << plist[ii*ncoefs+kk] << endl;
	  }
*/
	  cout << "end of compute pi ii= "<< ii << endl;
	  for(neighbor=0;neighbor<numneighs[ii];neighbor++){

		  x = rij[idpair][0];
		  y = rij[idpair][1];
		  z = rij[idpair][2];
		  idpair++;


		  r = sqrt(x*x + y*y + z*z);
		  if(r<pow(10,-8)) continue;

		  totali=nmax*numYlms;
		  for(ti=0;ti<totali;ti++) clist[ti]={0.0,0.0};
		  totali=nmax*numYlms*3;
		  for(int tn=0;tn<totali;tn++) dclist[tn]={0.0,0.0};

//////////************** for debug ********////////
//                 z=1.658;
////////////////////////////////////////////////////
		  compute_carray_wD(x, y, z, r, alpha, rcut, nmax, lmax, w,nmax,nmax,
												clist,nmax,numYlms,dclist,nmax,numYlms,3);
/*
		  cout << "x y z r alpha rcut nmax lmax "<<x<<" "<<y<<" "<<z<<" "<<r<<" "<<alpha<<" "<<rcut<<" "<<nmax<<" "<<lmax<<endl;
		  cout << " w " << endl;
		  for(int kk=0;kk<nmax*nmax;kk++){
			  cout << w[kk] << " ";
		  }
		  cout << endl;
*/
		  cout.precision(8);
	      cout <<std::scientific;
/*
		  cout << " clist " << endl;
		  for(int kk=0;kk<nmax*numYlms;kk++){
			  cout << clist[kk] << " ";
			  if(kk%2==1) cout << endl;
		  }
		  cout << endl;
*/


//		  cout << "end of compute_carray_wD ii= "<< ii << " neighbor "<<neighbor << " idpair " << idpair << endl;
		  totali=nmax*numYlms*3;
		  for(int tn=0;tn<totali;tn++){
			  dclist[tn] = dclist[tn]*double(weight);
		  }
/*
		  cout << " dclist " << endl;
		  for(int kk=0;kk<nmax*numYlms*3;kk++){
			  cout << dclist[kk] << " ";
			  if(kk%2==1 && kk%3 ==1) cout << endl;
			  if(kk%2==0 && kk%3 ==1) cout << endl;
			  else if(kk%3==2) cout << endl;
		  }
		  cout << endl;
*/
		  totali=numps*3;
		  for(ti=0;ti<totali;ti++) tempdp[ti]={0.0,0.0};
/*
		  cout << " clistot " << endl;
		  for(int kk=0;kk<nmax*numYlms;kk++){
			  cout << clisttot[kk] << " ";
			  if(kk%2==1) cout << endl;
		  }
		  cout << endl;
*/
	      compute_dpidrj(nmax, lmax, clisttot,nmax,numYlms, dclist,nmax,numYlms,3,
	                                           tempdp,numps,3);

/*
		  cout << "ii tempdp x y z "<<ii<<" " <<x<<" " << y << " " << z <<  endl;
		  for(int kk=0;kk<numps*3;kk++){
			  cout << tempdp[kk] << " ";
			  if(kk%2==1 && kk%3 ==1) cout << endl;
			  if(kk%2==0 && kk%3 ==1) cout << endl;
			  else if(kk%3==2) cout << endl;
		  }
		  cout << endl;
		  cout << "end of compute_dpidrj ii= "<< ii << " neighbor "<<neighbor << " idpair " << idpair << endl;
*/

		  for(int tn=0;tn<totali;tn++){
			  dplist[((idpair-1)*(numps*3))+tn] += tempdp[tn];
		  }


	  }

	}


	totali=natoms*ncoefs;
	for(i=0;i<totali;i++){
		 plist_r[i]=real(plist[i]);
		 plist_i[i]=imag(plist[i]);
	}
	print_2d("in compute_force plist_r",plist_r,natoms,ncoefs);

	totali=npairs*ncoefs*3;
	for(i=0;i<totali;i++){
		 dplist_r[i]=real(dplist[i]);
		 dplist_i[i]=imag(dplist[i]);
	}
	cout <<"npairs ncoefs numps "<<npairs <<" "<<ncoefs<<" "<<numps<<endl;
//	print_3d("dplist_r",dplist_r,npairs,ncoefs,3,0);

	delete tempdp;
	delete plist;
	delete dplist;
	return;



}
