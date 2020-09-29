#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>

#define K 5 //近邻数k
typedef float type;
#define nfolds 5

//动态创建二维数组
type **createarray(int n,int m) //n行m列，4177行9列
{
    int i;
    type **array;
    array=(type **)malloc(n*sizeof(type *));
    array[0]=(type *)malloc(n*m*sizeof(type));
    for(i=1;i<n;i++) array[i]=array[i-1]+m;
    return array;
}

//读取数据，要求首行格式为 N=数据量,D=维数
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

    *array=createarray(*n,*d);   //创建一个n*d的矩阵
    *karray=createarray(2,K); //2行K列的

    for(i=0;i<*n;i++)
        for(j=0;j<*d;j++)
            fscanf(fp,"%f",&(*array)[i][j]);    //读取数据

    for(i=0;i<2;i++)
        for(j=0;j<K;j++)
            (*karray)[i][j]=9999.0;    //默认的最大值

    if(fclose(fp)){
        fprintf(stderr,"can not close data.txt");
    }
    //在读取的时候，karray就出现。第一行存放到K个最近邻点的距离，第二行存放训练集的内容
}

//计算欧氏距离
type computedistance(int n,type *avector,type *bvector)
{
    int i;
    type dist=0.0;
    for(i=0;i<n;i++)
        dist+=pow(avector[i]-bvector[i],2); //求多维度对应差的平方根，test的data和每一个array行
    return sqrt(dist);
}

//冒泡排序
void bublesort(int n,type **a,int choice)  //用到K,karray，0/1
{
    int i,j;
    type k; //
    for(j=0;j<n;j++) //n是k
        for(i=0;i<n-j-1;i++){
            if(0==choice){//小在前面
                if(a[0][i]>a[0][i+1]){
                    k=a[0][i];
                    a[0][i]=a[0][i+1];
                    a[0][i+1]=k;
                    k=a[1][i];
                    a[1][i]=a[1][i+1];
                    a[1][i+1]=k;
                }
            }
            else if(1==choice){ //也是小的在前面，根据第2行排序
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

//统计有序表中的元素个数
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

    return value; //返回max
}

int main()
{   //动态分配的数组最后可能要free掉
    int i,j,k,c;
    int scores1=0;
    int scores2=0;
    int scores3=0;
    int scores4=0;
    int scores5=0;
    //用到的参数i,j,k,K
    int D,N,NN,N1;    //N行D列
    type **array=NULL;  //数据数组,N行D列，3342*9,N和D
    type **karray=NULL;
    type **train=NULL;
    type **Array=NULL;//预测集
    type *predict; //最终结果
    type dist,maxdist; //p是最终比率

    loaddata(&N,&D,&array,&karray);
    NN=N/nfolds;  //测试集的长度
    N1=N-NN;  //训练集的长度,原来的N--》N1
    train=createarray(N1,D);
    Array=createarray(NN,D);
    predict=(type *)malloc(NN*sizeof(type));

    srand(1); //随机种子,1是42,2是34

    //随机数生成第一个fold的训练集和测试集
    int t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//前D-1个数，预测最后一个数字

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:这么多行为什么s算前k行，而且还是排序前的
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //算出D-1维度点的距离，与第i行的距离放在第i列
            karray[1][i]=train[i][D-1]; //第二行是训练集的最后一个数字
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //初始化k近邻数组的距离最大值

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j后元素复制到后一位，为插入做准备
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //插入到j位置
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //不比较karray后续元素
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //前几名决出来后第二行就重要了

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores1=scores1+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //测试集行遍历结束
    float Accuracy1 =(scores1/835.0)*100;
    printf("scores1: %d,Accuracy1:%0.f %% \n",scores1,Accuracy1);

    //随机数生成第二个测试集

    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//前D-1个数，预测最后一个数字

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:这么多行为什么s算前k行，而且还是排序前的
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //算出D-1维度点的距离，与第i行的距离放在第i列
            karray[1][i]=train[i][D-1]; //第二行是训练集的最后一个数字
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //初始化k近邻数组的距离最大值

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j后元素复制到后一位，为插入做准备
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //插入到j位置
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //不比较karray后续元素
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //前几名决出来后第二行就重要了

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores2=scores2+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //测试集行遍历结束
    float Accuracy2 =(scores2/835.0)*100;
    printf("scores2: %d,Accuracy2:%0.f %% \n",scores2,Accuracy2);


    //随机生成第3组测试集和训练集

    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//前D-1个数，预测最后一个数字

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:这么多行为什么s算前k行，而且还是排序前的
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //算出D-1维度点的距离，与第i行的距离放在第i列
            karray[1][i]=train[i][D-1]; //第二行是训练集的最后一个数字
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //初始化k近邻数组的距离最大值

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j后元素复制到后一位，为插入做准备
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //插入到j位置
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //不比较karray后续元素
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //前几名决出来后第二行就重要了

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores3=scores3+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //测试集行遍历结束
    float Accuracy3 =(scores3/835.0)*100;
    printf("scores3: %d,Accuracy3:%0.f %% \n",scores3,Accuracy3);


    //随机生成第四组测试集和训练集
    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//前D-1个数，预测最后一个数字

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:这么多行为什么s算前k行，而且还是排序前的
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //算出D-1维度点的距离，与第i行的距离放在第i列
            karray[1][i]=train[i][D-1]; //第二行是训练集的最后一个数字
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //初始化k近邻数组的距离最大值

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j后元素复制到后一位，为插入做准备
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //插入到j位置
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //不比较karray后续元素
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //前几名决出来后第二行就重要了

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores4=scores4+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //测试集行遍历结束
    float Accuracy4 =(scores4/835.0)*100;
    printf("scores4: %d,Accuracy4:%0.f %% \n",scores4,Accuracy4);

    //生成第5组训练集和测试集
    t=0;
    while (t < NN)
    {
        int i=rand()%11*(N-t-1)/10;
        for( j = 0; j < D; j++){
           Array[t][j] = array[i][j];
        }
          for (j = i; j < (N-t-1); j++)//将随机选中的i行删除
          {
            array[j] = array[j + 1];
          }
           t++;
    }
    for(i=0;i<N1;i++){
      for(j=0;j<D;j++){
        train[i][j]=array[i][j];
      }
    }//前D-1个数，预测最后一个数字

    for(c=0;c<NN;c++){
        for(i=0;i<K;i++){
            if(K>N1) exit(-1);  //?1:这么多行为什么s算前k行，而且还是排序前的
            karray[0][i]=computedistance(D-1,Array[c],train[i]); //算出D-1维度点的距离，与第i行的距离放在第i列
            karray[1][i]=train[i][D-1]; //第二行是训练集的最后一个数字
        }

        bublesort(K,karray,0);
    //for(i=0;i<K;i++)    printf("after bublesort in first karray:%6.2f  %6.0f\n",karray[0][i],karray[1][i]);
        maxdist=karray[0][K-1]; //初始化k近邻数组的距离最大值

       for(i=K;i<N1;i++){ //i=3--6
        dist=computedistance(D-1,Array[c],train[i]);
        if(dist<maxdist){
            for(j=0;j<K;j++){
                if(dist<karray[0][j]){
                    for(k=K-1;k>j;k--){ //j后元素复制到后一位，为插入做准备
                        karray[0][k]=karray[0][k-1];
                        karray[1][k]=karray[1][k-1];
                    }
                    karray[0][j]=dist;  //插入到j位置
                    karray[1][j]=train[i][D-1];
                    //printf("i:%d  karray:%6.2f %6.0f\n",i,karray[0][j],karray[1][j]);
                    break;  //不比较karray后续元素
                }
            }
       }
        maxdist=karray[0][K-1];
        //printf("i:%d  maxdist:%6.2f\n",i,maxdist);
      }
      bublesort(K,karray,1); //前几名决出来后第二行就重要了

      predict[c]=orderedlist(K,karray[1]);

      if(predict[c]==Array[c][D-1]){
            scores5=scores5+1;
        }
        //printf("\n scores:%d \n",scores);
    }
     //测试集行遍历结束
    float Accuracy5 =(scores5/835.0)*100;
    printf("scores5: %d,Accuracy5:%0.f %% \n",scores5,Accuracy5);

    printf("mean_accuracy:%0.f %% \n",(Accuracy1+Accuracy2+Accuracy3+Accuracy4+Accuracy5)/5);
    return 0;

}

