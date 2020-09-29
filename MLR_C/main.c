#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float type;
#define nfold 5
#define split 0.2


type **createarray(int n,int m) //������ά����
{
    int i;
    type **array;
    array=(type **)malloc(n*sizeof(type *));
    array[0]=(type *)malloc(m*n*sizeof(type));
    for(i=1;i<n;i++) array[i]=array[i-1]+m;
    return array;
}

void loaddata(int *n,int *d,type ***array)
{
    int i,j;
    FILE *fp;
    if((fp=fopen("MLR.txt","r"))==NULL){
        fprintf(stderr,"can not open data.txt!\n");
    }
    if(fscanf(fp,"N=%d,D=%d",n,d)!=2){
        fprintf(stderr,"reading error!\n");
    }

    *array=createarray(*n,*d);   //����һ��n*d�ľ���

    for(i=0;i<*n;i++)
        for(j=0;j<*d;j++)
            fscanf(fp,"%f",&(*array)[i][j]);    //��ȡ����

    if(fclose(fp)){
        fprintf(stderr,"can not close data.txt");
    }
    //�ڶ�ȡ��ʱ��karray�ͳ��֡���һ�д�ŵ�K������ڵ�ľ��룬�ڶ��д��ѵ����������
}

double predict(type *row,type *coefficients,int D){ //��������һά��̬���飬����D

    double yhat;
    int i;
    yhat=coefficients[0];
    for(i=0;i<D-1;i++)
    {
        yhat+=coefficients[i+1]*row[i];
    }
    return yhat;
}

//����ˢ��ϵ��
type *coefficient_sgd(type **array,type l_rate,type n_epoch,int D,int size2)//ѵ������ѵ���ٶȣ�ѵ������
{
    double yhat,error;
    int i,j,k;
    type *coef;
    coef=(type *)malloc(D*sizeof(type));
    for(i=0;i<D;i++){
      coef[i]=0.0;
    }   //��ʼ��ϵ������

    for(i=0;i<n_epoch;i++){
        for(j=0;j<size2;j++){
            yhat=predict(array[j],coef,D);
            error=yhat-array[j][D-1];
            coef[0]=coef[0]-l_rate*error;
            for(k=0;k<D-1;k++){
                coef[k+1]=coef[k+1]-l_rate*error*array[j][k];
            }
        }
    }
    return coef;
}

//���ز��Լ��Ľ������
type *linear_regression_sgd(type **train,type **test,double l_rate,double n_epoch,int D,int size1,int size2)
{
    double yyhat;
    int i;
    type *predictions;
    type *coeff;
    predictions=(type *)malloc(size1*sizeof(type));
    coeff=(type *)malloc(D*sizeof(type));

    coeff=coefficient_sgd(train,l_rate,n_epoch,D,size2);
    for(i=0;i<size1;i++){
        yyhat=predict(test[i],coeff,D);
        predictions[i]=yyhat;
    }

    return predictions;
}

double rmse_metric(type **test,type **train,double l_rate,double n_epoch,int D,int size1,int size2) //����Ԥ�⼯�Ͳ��Լ�
{
   double sumerror=0.0;
   double predictionerror=0.0;
   double meanerror;
   int i;
   type *Array;
   Array=(type *)malloc(size1*sizeof(type));
   Array=linear_regression_sgd(train,test,l_rate,n_epoch,D,size1,size2);

   for(i=0;i<size1;i++){
     predictionerror=Array[i]-test[i][D-1]; //�����һ�������бȽ�
     sumerror+=pow(predictionerror,2);
   }
   meanerror=sumerror/size1;

   return sqrt(meanerror);
}

int main()//��һ��foldΪ���Լ�������Ϊѵ����
{

    int i,j,k;
    int D,N;
    int size1,size2;

    double result1=0.0;
    double result2=0.0;
    double result3=0.0;
    double result4=0.0;
    double result5=0.0;

    double l_rate=0.01;
    double n_epoch=50;
    type **Sarray=NULL;//��ʼ����ֵ
    type **test=NULL;
    type **train=NULL;

    loaddata(&N,&D,&Sarray);//����һ�������ݿ�

   //����ѵ�����Ͳ��Լ�
    size1=split*N; //���Լ��Ĵ�С
    size2=N-size1;//size1ѵ������size2���Լ�
    train=createarray(size2,D);
    test=createarray(size1,D);

    //���ڶ�Ԫ���Ե�����֮��Ƚ϶�������˸���5�����������ֳ�5����ͬ�Ĳ��Լ����н�����֤
    //��һ��fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j];
    }
    for(i=0;i<size1;i++){
      for(k=0;k<4;k++){
       train[i*(nfold-1)+k]=Sarray[nfold*i+k+1]; //����������5��1234�����Ĵ���
      }
    }
    result1=rmse_metric(test,train,l_rate,n_epoch,D,size1,size2);
    printf("The first fold rmse: %f \n",result1);

    //�ڶ���fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j+1];
    }
    for(i=0;i<size1;i++){
       //nfold��5
       train[i*(nfold-1)]=Sarray[nfold*i]; //��������5��1234�����Ĵ���
       train[i*(nfold-1)+1]=Sarray[nfold*i+2];
       train[i*(nfold-1)+2]=Sarray[nfold*i+3];
       train[i*(nfold-1)+3]=Sarray[nfold*i+4];
    }
    result2=rmse_metric(test,train,l_rate,n_epoch,D,size1,size2);
    printf("The second fold rmse: %f \n",result2);

    //������fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j+2];
    }
    for(i=0;i<size1;i++){
       //nfold��5
       train[i*(nfold-1)]=Sarray[nfold*i]; //��������5��1234�����Ĵ���
       train[i*(nfold-1)+1]=Sarray[nfold*i+1];
       train[i*(nfold-1)+2]=Sarray[nfold*i+3];
       train[i*(nfold-1)+3]=Sarray[nfold*i+4];
    }
    result3=rmse_metric(test,train,l_rate,n_epoch,D,size1,size2);
    printf("The third fold rmse: %f \n",result3);

    //���ĸ�fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j+3];
    }
    for(i=0;i<size1;i++){
       //nfold��5
       train[i*(nfold-1)]=Sarray[nfold*i]; //��������5��1234�����Ĵ���
       train[i*(nfold-1)+1]=Sarray[nfold*i+1];
       train[i*(nfold-1)+2]=Sarray[nfold*i+2];
       train[i*(nfold-1)+3]=Sarray[nfold*i+4];
    }
    result4=rmse_metric(test,train,l_rate,n_epoch,D,size1,size2);
    printf("The fourth fold rmse: %f \n",result4);

    //�����fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j+4];
    }
    for(i=0;i<size1;i++){
       //nfold��5
       train[i*(nfold-1)]=Sarray[nfold*i]; //��������5��1234�����Ĵ���
       train[i*(nfold-1)+1]=Sarray[nfold*i+1];
       train[i*(nfold-1)+2]=Sarray[nfold*i+2];
       train[i*(nfold-1)+3]=Sarray[nfold*i+3];
    }
    result5=rmse_metric(test,train,l_rate,n_epoch,D,size1,size2);
    printf("The fifth fold rmse: %f \n",result5);
    printf("The mean RMSE:%f",(result1+result2+result3+result4+result5)/nfold);

    return 0;
}
