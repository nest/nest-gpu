clear all; close all;
data=load('test_neuron_groups_voltage.dat');
t=data(:,1);
n1=data(:,2);
n2=data(:,3);
n3=data(:,4);
n4=data(:,5);
n5=data(:,6);
n6=data(:,7);
n7=data(:,8);
%
data1=load('test_neuron_group_record.dat');
t1=data1(:,1);
v1=data1(:,2);
v2=data1(:,3);
v3=data1(:,4);
v4=data1(:,5);
v5=data1(:,6);
v6=data1(:,7);
v7=data1(:,8);
%

figure(1);
plot(t,n1);
hold on;
plot(t1,v1);

figure(2);
plot(t,n2);
hold on;
plot(t1,v2);

figure(3);
plot(t,n3);
hold on;
plot(t1,v3);

figure(4);
plot(t,n4);
hold on;
plot(t1,v4);

figure(5);
plot(t,n5);
hold on;
plot(t1,v5);

figure(6);
plot(t,n6);
hold on;
plot(t1,v6);

figure(7);
plot(t,n7);
hold on;
plot(t1,v7);
