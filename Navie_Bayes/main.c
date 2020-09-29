#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef float type;
#define nfold 5
#define N 150
#define D 5
#define nclass 3

type **createarray(int n,int m) //������ά����
{
    int i;
    type **array;
    array=(type **)malloc(n*sizeof(type *));
    array[0]=(type *)malloc(m*n*sizeof(type)); //���ܳ�����
    for(i=1;i<n;i++) array[i]=array[i-1]+m;
    return array;
}

void loaddata(int n,int d,type ***array)
{
    int i,j;
    FILE *fp;
    if((fp=fopen("IRIS.txt","r"))==NULL){
        fprintf(stderr,"can not open data.txt!\n");
    }

    *array=createarray(n,d);   //����һ��n*d�ľ���

    for(i=0;i<n;i++)
        for(j=0;j<d;j++)
            fscanf(fp,"%f",&(*array)[i][j]);    //��ȡ����

    if(fclose(fp)){
        fprintf(stderr,"can not close data.txt");
    }
    //�ڶ�ȡ��ʱ��karray�ͳ��֡���һ�д�ŵ�K������ڵ�ľ��룬�ڶ��д��ѵ����������
}

double meanlist(type *array,int size) //һά��������鳤��
{
    double sum=0.0;
    int i;
    for(i=0; i<size; i++) //
    {
        sum += array[i];
    }
    return sum/size;
}

double stdev(type *array,int size) //sizeΪ���ݸ���
{                                    //������mean����
    double avg;
    int i;
    double variance=0.0;
    avg=meanlist(array,size);

    for(i=0; i<size; i++) //
    {
        variance+=pow(array[i]-avg,2);
    }
    variance=variance/(size-1);
    return sqrt(variance);
}

double calculate_probability(type x,double mean,double stdev) //��ֵ�ͱ�׼��ֱ�Ӵ��뷵��ֵ
{
    double exponent;
    double res;
    double pi=3.14159;
    exponent=exp(-(pow(x-mean,2)/(2*pow(stdev,2))));
    res=(1/(sqrt(2*pi)*stdev))*exponent;
    return res;
}

type **summarize_dataset(type **array,int d,int size) //����һ����ά���飬����һ��(D-1)*3������,size���ȣ�size2train������
{                                                               //size2:120,size 40 //���þ�ֵ�������
    type **summary=NULL;
    type **zhuanzhi=NULL;
    int i,j;
    summary=createarray(d,nclass); //5*3
    zhuanzhi=createarray(d,size); //������л����ľ�����Ϣ
    //������ת��
    for(i=0;i<size;i++){ //������ ,����40��D�е���
        for(j=0;j<d;j++){ //������
            zhuanzhi[j][i]=array[i][j];
        }
    }

    for(i=0;i<d-1;i++){ //���һ�е����ݲ�Ҫ
        summary[i][0]=meanlist(zhuanzhi[i],size);
        summary[i][1]=stdev(zhuanzhi[i],size);
        summary[i][2]=d;
    }
    return summary;
}

type **separate_by_class(type **array,int n,int d) //array��n��d�е�120*5,����ѵ������ʽ��
{
    type **separated=NULL;
    separated=createarray(n,d);
    type **separated1=NULL;
    separated1=createarray(n/3,d);
    type **separated2=NULL;
    separated2=createarray(n/3,d);
    type **separated3=NULL;
    separated3=createarray(n/3,d);

    //separated=array;
    int i;
    int t=0;
    int tt=0;
    int ttt=0;
    for(i=0;i<n;i++){
      if(array[i][d-1]==0){
        separated1[t]=array[i];
        t++;
      }
      if(array[i][d-1]==1){
        separated2[tt]=array[i];
        tt++;
      }
      if(array[i][d-1]==2){
        separated3[ttt]=array[i];
        ttt++;
      }
    }
    for(i=0;i<n/3;i++){
      separated[nclass*i]=separated1[i];
      separated[nclass*i+1]=separated2[i];
      separated[nclass*i+2]=separated3[i];
    }
    /*for(i=0;i<n;i++){
      for(t=0;t<d;t++){
        printf("%f",separated[i][t]);
      }
    }*/
    return separated;
}

 type *calculate_class_probabilities(type **array,type *row,int d,int n) //separated,row��test��ĳһ��
 {                                                                                 //����cal_p����,summarizez_dataset����
     int i;
     //int total_rows=N-N/nfold;
     type *probablity;
     probablity=(type *)malloc(nclass*sizeof(type));
     type **separated1=NULL;
     separated1=createarray(n/3,d);
     type **separated2=NULL;
     separated2=createarray(n/3,d);
     type **separated3=NULL;
     separated3=createarray(n/3,d);
     for(i=0;i<nclass;i++){
        probablity[i]=0.33;
     }
     for(i=0;i<n/3;i++){
      separated1[i]=array[nclass*i];
      separated2[i]=array[nclass*i+1];
      separated3[i]=array[nclass*i+2];
     }
     //��ʼ������
     type **summary=NULL;
     summary=createarray(d,nclass); //5*3

     //row���ǵ�һ����ǩ����
     summary=summarize_dataset(separated1,D,40);

     for(i=0;i<d-1;i++){
       probablity[0]=probablity[0]*calculate_probability(row[i],summary[i][0],summary[i][1]);
      // printf("%f \n",probablity[0]);
     }
     /*for(i=0;i<d-1;i++){
        for(j=0;j<nclass;j++){
            printf("%f",summary[i][j]);
        }
     }*/
     summary=summarize_dataset(separated2,D,40);
     for(i=0;i<d-1;i++){
       probablity[1]=probablity[1]*calculate_probability(row[i],summary[i][0],summary[i][1]);
       //printf("%f \n",probablity[1]);
     }

     summary=summarize_dataset(separated3,D,40);
     for(i=0;i<d-1;i++){
       probablity[2]=probablity[2]*calculate_probability(row[i],summary[i][0],summary[i][1]);
      // printf("%f \n",probablity[2]);
     }

    return probablity;
 }

double predict(type *array)
{
    int i;
    double result;
    double max;
    max=array[0];
    result=0;
    for(i=0;i<nclass;i++){
        if(array[i]>max){
            max=array[i];
            result=i;
        }
    }

//printf("%f",result);
    return result;
}

type *naive_bayes(type **train,type **test,int size2,int size1,int d) //120,30,5
{
    int i;
    type **Array=NULL;
    Array=createarray(size2,d);
    type *prediction;
    prediction=(type *)malloc(size1*sizeof(type));
    Array=separate_by_class(train,size2,d); //������������

    type *probablity;
    probablity=(type *)malloc(nclass*sizeof(type));
    for(i=0;i<size1;i++){
       probablity=calculate_class_probabilities(Array,test[i],d,size2);
       prediction[i]=predict(probablity);
       //printf("%d",prediction[i]);
    }
    return prediction;
}

int evaluate_algorithm(type **train,type **test,int size2,int size1,int d)
{
    int i;
    int scores=0;
    type *ppre;
    ppre=(type *)malloc(size1*sizeof(type));

    ppre=naive_bayes(train,test,size2,size1,d);
    for(i=0;i<size1;i++){
        if(ppre[i]==test[i][d-1]){
            scores+=1;
        }
    }
    //printf("%d",ppre);
    return scores;
}

int main()
{
    int i,j,k;
    int scores;
    double accuracy1;
    double accuracy2;
    double accuracy3;
    double accuracy4;
    double accuracy5;
    //int D,N;
    int size1,size2;
    type **Sarray=NULL;
    type **test=NULL;
    type **train=NULL;

    loaddata(N,D,&Sarray);
    size1=N/nfold;
    size2=N-size1;
    train=createarray(size2,D);
    test=createarray(size1,D);
    //����ѵ������Ԥ�⼯
    //�����һ�б�Ϊ����
    for(i=0;i<N;i++){
        Sarray[i][D-1]=(int)Sarray[i][D-1];
    }

    //��һ��fold
    for(j=0;j<size1;j++){
      test[j]=Sarray[nfold*j];
    }
    for(i=0;i<size1;i++){
      for(k=0;k<4;k++){
       train[i*(nfold-1)+k]=Sarray[nfold*i+k+1]; //����������5��1234�����Ĵ���
      }
    }
    scores=evaluate_algorithm(train,test,size2,size1,D);
    accuracy1=(float)scores/(float)size1;
    printf("accuracy1: %f \n",accuracy1);

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
    scores=evaluate_algorithm(train,test,size2,size1,D);
    accuracy2=(float)scores/(float)size1;
    printf("accuracy2: %f \n",accuracy2);

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
    scores=evaluate_algorithm(train,test,size2,size1,D);
    accuracy3=(float)scores/(float)size1;
    printf("accuracy3: %f \n",accuracy3);

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
    scores=evaluate_algorithm(train,test,size2,size1,D);
    accuracy4=(float)scores/(float)size1;
    printf("accuracy4: %f \n",accuracy4);

    //��5��fold
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
    scores=evaluate_algorithm(train,test,size2,size1,D);
    accuracy5=(float)scores/(float)size1;
    printf("accuracy5: %f \n",accuracy5);

    printf("mean_accuracy: %f \n",(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/nfold);
    return 0;
}
