#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>

#define K 5 //������k
typedef float type;
#define nfolds 5

//��̬������ά����
type **createarray(int n,int m) //n��m�У�4177��9��
{
    int i;
    type **array;
    array=(type **)malloc(n*sizeof(type *));
    array[0]=(type *)malloc(n*m*sizeof(type));
    for(i=1;i<n;i++) array[i]=array[i-1]+m;
    return array;
}

//��ȡ���ݣ�Ҫ�����и�ʽΪ N=������,D=ά��
void loaddata(int *n,int *d,type ***array,type ***karray)
{
    int i,j;
    FILE *fp;
    if((fp=fopen("train1.txt","r"))==NULL){
        fprintf(stderr,"can not open data.txt!\n");
    }
    if(fscanf(fp,"N=%d,D=%d",n,d)!=2){
        fprintf(stderr,"reading error!\n");
    }

    *array=createarray(*n,*d);   //����һ��n*d�ľ���
    *karray=createarray(2,K); //2��K�е�

    for(i=0;i<*n;i++)
        for(j=0;j<*d;j++)
            fscanf(fp,"%f",&(*array)[i][j]);    //��ȡ����

    for(i=0;i<2;i++)
        for(j=0;j<K;j++)
            (*karray)[i][j]=9999.0;    //Ĭ�ϵ����ֵ

    if(fclose(fp)){
        fprintf(stderr,"can not close data.txt");
    }
    //�ڶ�ȡ��ʱ��karray�ͳ��֡���һ�д�ŵ�K������ڵ�ľ��룬�ڶ��д��ѵ����������
}

//����ŷ�Ͼ���
type computedistance(int n,type *avector,type *bvector)
{
    int i;
    type dist=0.0;
    for(i=0;i<n;i++)
        dist+=pow(avector[i]-bvector[i],2); //���ά�ȶ�Ӧ���ƽ������test��data��ÿһ��array��
    return sqrt(dist);
}

//ð������
void bublesort(int n,type **a,int choice)  //�õ�K,karray��0/1
{
    int i,j;
    type k; //
    for(j=0;j<n;j++) //n��k
        for(i=0;i<n-j-1;i++){
            if(0==choice){//С��ǰ��
                if(a[0][i]>a[0][i+1]){
                    k=a[0][i];
                    a[0][i]=a[0][i+1];
                    a[0][i+1]=k;
                    k=a[1][i];
                    a[1][i]=a[1][i+1];
                    a[1][i+1]=k;
                }
            }
            else if(1==choice){ //Ҳ��С����ǰ�棬���ݵ�2������
                if(a[1][i]>a[1][i+1]){
                    k=a[0][i];
                    a[0][i]=a[0][i+1];
                    a[0][i+1]=k;
                    k=a[1][i];
                    a[1][i]=a[1][i+1];
                    a[1][i+1]=k;
                }
            }
        }
}

//ͳ��������е�Ԫ�ظ���
type orderedlist(int n,type *list) //K,array[1]
{
    int i,count=1,maxcount=1;
    type value; //
    for(i=0;i<(n-1);i++) {
        if(list[i]!=list[i+1]) {

            if(count>maxcount){
                maxcount=count;
                value=list[i];
                count=1;
            }
        }
        else
            count++;
    }
    if(count>maxcount){
            maxcount=count;
            value=list[n-1];
    }

    return value; //����max
}

int main()
{   //��̬���������������Ҫfree��
    int i,j,k,c;
    int scores1=0;
    int scores2=0;
    int scores3=0;
    int scores4=0;
    int scores5=0;
    //�õ��Ĳ���i,j,k,K
    int D,N,NN,N1;    //N��D��
    type **array=NULL;  //��������,N��D�У�3342*9,N��D
    type **karray=NULL;
    type **train=NULL;
    type **Array=NULL;//Ԥ�⼯
    type *predict; //���ս��
    type dist,maxdist; //p�����ձ���

    loaddata(&N,&D,&array,&karray);
    NN=N/nfolds;  //���Լ��ĳ���
    N1=N-NN;  //ѵ�����ĳ���,ԭ����N--��N1
    train=createarray(N1,D);
    Array=createarray(NN,D);
    predict=(type *)malloc(NN*sizeof(type));

    srand(1); //�������,1��42,2��34

    //��������ɵ�һ��fold��ѵ�����Ͳ��Լ�
    int t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//�����ѡ�е�i��ɾ��
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//ǰD-1������Ԥ�����һ������

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:��ô����Ϊʲôs��ǰk�У����һ�������ǰ��
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //���D-1ά�ȵ�ľ��룬���i�еľ�����ڵ�i��
            karray[1][i]=train[i][D-1]; //�ڶ�����ѵ���������һ������
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //��ʼ��k��������ľ������ֵ

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j��Ԫ�ظ��Ƶ���һλ��Ϊ������׼��
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //���뵽jλ��
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //���Ƚ�karray����Ԫ��
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //ǰ������������ڶ��о���Ҫ��

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores1=scores1+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //���Լ��б�������
    float Accuracy1 =(scores1/835.0)*100;
    printf("scores1: %d,Accuracy1:%0.f %% \n",scores1,Accuracy1);

    //��������ɵڶ������Լ�

    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//�����ѡ�е�i��ɾ��
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//ǰD-1������Ԥ�����һ������

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:��ô����Ϊʲôs��ǰk�У����һ�������ǰ��
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //���D-1ά�ȵ�ľ��룬���i�еľ�����ڵ�i��
            karray[1][i]=train[i][D-1]; //�ڶ�����ѵ���������һ������
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //��ʼ��k��������ľ������ֵ

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j��Ԫ�ظ��Ƶ���һλ��Ϊ������׼��
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //���뵽jλ��
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //���Ƚ�karray����Ԫ��
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //ǰ������������ڶ��о���Ҫ��

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores2=scores2+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //���Լ��б�������
    float Accuracy2 =(scores2/835.0)*100;
    printf("scores2: %d,Accuracy2:%0.f %% \n",scores2,Accuracy2);


    //������ɵ�3����Լ���ѵ����

    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//�����ѡ�е�i��ɾ��
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//ǰD-1������Ԥ�����һ������

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:��ô����Ϊʲôs��ǰk�У����һ�������ǰ��
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //���D-1ά�ȵ�ľ��룬���i�еľ�����ڵ�i��
            karray[1][i]=train[i][D-1]; //�ڶ�����ѵ���������һ������
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //��ʼ��k��������ľ������ֵ

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j��Ԫ�ظ��Ƶ���һλ��Ϊ������׼��
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //���뵽jλ��
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //���Ƚ�karray����Ԫ��
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //ǰ������������ڶ��о���Ҫ��

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores3=scores3+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //���Լ��б�������
    float Accuracy3 =(scores3/835.0)*100;
    printf("scores3: %d,Accuracy3:%0.f %% \n",scores3,Accuracy3);


    //������ɵ�������Լ���ѵ����
    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//�����ѡ�е�i��ɾ��
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//ǰD-1������Ԥ�����һ������

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:��ô����Ϊʲôs��ǰk�У����һ�������ǰ��
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //���D-1ά�ȵ�ľ��룬���i�еľ�����ڵ�i��
            karray[1][i]=train[i][D-1]; //�ڶ�����ѵ���������һ������
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //��ʼ��k��������ľ������ֵ

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j��Ԫ�ظ��Ƶ���һλ��Ϊ������׼��
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //���뵽jλ��
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //���Ƚ�karray����Ԫ��
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //ǰ������������ڶ��о���Ҫ��

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores4=scores4+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //���Լ��б�������
    float Accuracy4 =(scores4/835.0)*100;
    printf("scores4: %d,Accuracy4:%0.f %% \n",scores4,Accuracy4);

    //���ɵ�5��ѵ�����Ͳ��Լ�
    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//�����ѡ�е�i��ɾ��
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//ǰD-1������Ԥ�����һ������

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:��ô����Ϊʲôs��ǰk�У����һ�������ǰ��
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //���D-1ά�ȵ�ľ��룬���i�еľ�����ڵ�i��
            karray[1][i]=train[i][D-1]; //�ڶ�����ѵ���������һ������
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //��ʼ��k��������ľ������ֵ

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j��Ԫ�ظ��Ƶ���һλ��Ϊ������׼��
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //���뵽jλ��
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //���Ƚ�karray����Ԫ��
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //ǰ������������ڶ��о���Ҫ��

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores5=scores5+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //���Լ��б�������
    float Accuracy5 =(scores5/835.0)*100;
    printf("scores5: %d,Accuracy5:%0.f %% \n",scores5,Accuracy5);

    printf("mean_accuracy:%0.f %% \n",(Accuracy1+Accuracy2+Accuracy3+Accuracy4+Accuracy5)/5);
    return 0;

}

