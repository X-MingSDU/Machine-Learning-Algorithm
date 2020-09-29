#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>   //生成随机数

typedef float type;
#define split 0.6

//动态创建二维数组
type **createarray(int n,int m) //
{
    int i;
    type **array;
    array=(type **)malloc(n*sizeof(type *));
    array[0]=(type *)malloc(n*m*sizeof(type));
    for(i=1;i<n;i++) array[i]=array[i-1]+m;
    return array;
}

//均值函数
double mean_x(type **array,int N)
{
    int i;
    double sum=0;
    double meanx;
    for(i = 0; i < N; i++) {
        sum=sum+array[i][0];
    }
    meanx=sum/N;
    return meanx;
}

double mean_y(type **array,int N)
{
    int i;
    double sum=0;
    double meany;
    for(i = 0;i<N; i++) {
        sum=sum+array[i][1];
    }
    meany=sum/N;
    return meany;
}

//方差函数
double variance_x(double m,type **array,int N) //读入均值和训练集参数
{
  int i;
  double varx=0.0;
  for(i=0;i<N;i++){
    varx+=pow((array[i][0]-m),2);
  }
  return varx;
}

double variance_y(double m,type **array,int N) //读入均值和训练集参数
{
  int i;
  double vary=0.0;
  for(i=0;i<N;i++){
    vary+=pow((array[i][1]-m),2);
  }
  return vary;
}

//协方差函数,读入mean
double covariance(double a,double b,type **array,int N) //ab分别为x和y的均值
{
    int i;
    double covar=0.0;
    for(i=0;i<N;i++)
    {
     covar+=(array[i][0]-a)*(array[i][1]-b);
    }
    return covar;
}

//求系数
/*
double coefficient(double a,double b,double c,double d) //covar、varx、ymean、xmean
{
  double b0,b1;
  double coef[2];
  b1=a/b;
  b0=c-(b1*d);
  coef[0]=b0;
  coef[1]=b1;

  return coef;
}*/

//预测函数,写在main函数里面

void loadfile(int *n,int *d,type ***array)
{
    int i,j;
    FILE *fp;
    if((fp=fopen("SLR.txt","r"))==NULL){
        fprintf(stderr,"can not open data.txt!\n");
    }
    if(fscanf(fp,"N=%d,D=%d",n,d)!=2){
        fprintf(stderr,"reading error!\n");
    }

    *array=createarray(*n,*d);   //创建一个n*d的矩阵

    for(i=0;i<*n;i++)
        for(j=0;j<*d;j++)
            fscanf(fp,"%f",&(*array)[i][j]);    //读取数据

    if(fclose(fp)){
        fprintf(stderr,"can not close data.txt");
    }//在读取的时候，karray就出现。第一行存放到K个最近邻点的距离，第二行存放训练集的内容
}

int main()
{
    int i,j;
    //用到的参数i,j,k,K
    int D,N,size1,size2;
    type **test=NULL;
    type **Train=NULL;
    type **train=NULL; //数据数组,N行D列，3342*9,N和D
    //type **Array=NULL;
    loadfile(&N,&D,&Train);
    size2=N*split;
    size1=N-size2;
    test=createarray(size1,D);
    train=createarray(size2,D);
    //运用随机数生成一个test
    srand(2); //随机种子,1是42,2是34
    int t=0;
    while (t < size1)
    {
        //int i;
        //i = (int)((double)(row-t-1)) * rand() / (double)(RAND_MAX); //网上找的?
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           test[t][j] = Train[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            Train[j] = Train[j + 1];
          }
           t++;

    }
    for(i=0;i<size2;i++){
      for(j=0;j<D;j++){
        train[i][j]=Train[i][j];
      }
    }

    type *predict;
    double rmse;
    double meanx,meany,covar,varx;
    double b0,b1;
     //训练集//测试集
    predict=(type *)malloc(size1*sizeof(type));

    meanx=mean_x(train,size2);
    meany=mean_y(train,size2);
    varx=variance_x(meanx,train,size2);
    covar=covariance(meanx,meany,train,size2);
    b1=covar/varx;
    b0=meany-b1*meanx;
    double pre=0.0;
    for(i=0;i<size1;i++)
    {
       predict[i]=b0+test[i][0]*b1;

    }
    //计算标准差
    type sumerror=0.0;
    for(j=0;j<size1;j++)
    {
        sumerror+=pow(predict[j]-test[j][1],2);
    }
    rmse=sumerror/size1;
    rmse=sqrt(rmse);
    printf("RMSE: %f \n",rmse);

    return 0;
}
