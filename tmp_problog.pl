% Auto-generated ProbLog rules from RuleNetwork branches

threshold(t0_0,5.652177022).
threshold(t0_1,0.6492969138).
threshold(t0_2,7.408786808).
threshold(t0_4,4.393862796).
threshold(t0_5,3.708582335).
threshold(t0_7,0.7613449645).
threshold(t0_8,4.630233898).
threshold(t0_10,5.549898764).
threshold(t0_11,2.159740183).
threshold(t0_15,2.761699992).
threshold(t0_17,6.09553307).
threshold(t0_18,5.076117453).
threshold(t0_22,5.110414337).
threshold(t0_25,1.948632731).
threshold(t0_26,1.823291934).
threshold(t0_29,5.023846039).
threshold(t0_32,1.77176573).
threshold(t1_0,1.968039274).
threshold(t1_1,6.030408457).
threshold(t1_3,1.212115282).
threshold(t1_4,7.028174043).
threshold(t1_5,3.042389973).
threshold(t1_6,5.074765021).
threshold(t1_7,3.981413016).
threshold(t1_9,4.512651819).
threshold(t1_12,0.985974344).
threshold(t1_15,4.554281774).
threshold(t1_18,1.730802495).
threshold(t1_21,2.718945495).
threshold(t1_22,5.822048677).
threshold(t1_25,4.174416611).
threshold(t1_28,1.412243024).
threshold(t2_0,6.13681925).
threshold(t2_1,4.599292598).
threshold(t2_2,1.564809484).
threshold(t2_4,4.70668735).
threshold(t2_6,3.513796159).
threshold(t2_7,0.4641412279).
threshold(t2_10,4.292886851).
threshold(t2_12,1.920526685).
threshold(t2_13,1.031626797).
threshold(t2_14,4.704729852).
threshold(t2_15,5.639545098).
threshold(t2_16,2.549513593).
threshold(t2_18,1.717889325).
threshold(t2_20,5.800258127).
threshold(t2_23,2.900107153).
threshold(t2_27,1.614119514).
threshold(t2_31,1.836384565).
threshold(t2_33,2.962185078).
threshold(t2_36,1.767427861).
threshold(t3_0,4.872928289).
threshold(t3_1,0.5934598824).
threshold(t3_2,6.975249466).
threshold(t3_4,2.602285179).
threshold(t3_5,6.625229426).
threshold(t3_7,3.196607648).
threshold(t3_8,1.535886185).
threshold(t3_10,1.968534121).
threshold(t3_11,5.614591271).
threshold(t3_15,2.200151902).
threshold(t3_18,5.183180611).
threshold(t3_19,4.987307369).
threshold(t3_22,5.892258284).
threshold(t3_25,1.612053442).
threshold(t3_26,6.116269615).
threshold(t3_29,2.101430101).
threshold(t4_0,0.4036575052).
threshold(t4_2,6.44573226).
threshold(t4_3,4.273657641).
threshold(t4_4,6.807475858).
threshold(t4_5,5.527132052).
threshold(t4_6,4.995446499).
threshold(t4_7,1.37882288).
threshold(t4_8,6.140451424).
threshold(t4_9,1.893363946).
threshold(t4_10,7.630854566).
threshold(t4_11,5.764233539).
threshold(t4_13,3.063528275).
threshold(t4_15,2.496616911).
threshold(t4_18,2.01884499).
threshold(t4_21,3.056669902).
threshold(t4_24,3.437894493).
threshold(t4_27,5.845095607).
threshold(t4_30,1.52049784).
threshold(t4_33,4.852697238).
threshold(t4_36,5.129811036).
threshold(t4_40,4.635853415).
threshold(t4_41,1.613824486).
threshold(t5_0,1.321920761).
threshold(t5_1,0.6874098076).
threshold(t5_2,6.124378775).
threshold(t5_5,2.445116173).
threshold(t5_7,4.893669708).
threshold(t5_9,1.610879694).
threshold(t5_10,1.79824636).
threshold(t5_11,2.531335017).
threshold(t5_14,4.648670243).
threshold(t5_17,2.381801263).
threshold(t6_0,5.296354102).
threshold(t6_1,4.298592518).
threshold(t6_3,2.352810359).
threshold(t6_4,4.966805732).
threshold(t6_7,1.460356714).
threshold(t6_8,1.686961224).
threshold(t6_9,5.085103185).
threshold(t6_10,1.875990977).
threshold(t6_16,5.013149585).
threshold(t7_0,1.055024205).
threshold(t7_1,0.9007414445).
threshold(t7_2,4.844717674).
threshold(t7_3,1.151904029).
threshold(t7_4,4.968072713).
threshold(t7_8,2.158642628).
threshold(t7_9,2.234638057).
threshold(t7_12,6.14728874).
threshold(t7_13,1.6806342).
threshold(t7_14,2.524810166).
threshold(t7_18,1.740370494).
threshold(t7_22,4.990644515).

branch_struct(b0, X) :- le(f0,t0_0,X), le(f3,t0_1,X).
branch_struct(b1, X) :- le(f0,t0_0,X), gt(f3,t0_1,X), le(f2,t0_4,X).
branch_struct(b2, X) :- le(f0,t0_0,X), gt(f3,t0_1,X), gt(f2,t0_4,X), le(f0,t0_22,X).
branch_struct(b3, X) :- le(f0,t0_0,X), gt(f3,t0_1,X), gt(f2,t0_4,X), gt(f0,t0_22,X).
branch_struct(b4, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), le(f2,t0_5,X), le(f3,t0_7,X).
branch_struct(b5, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), le(f2,t0_5,X), gt(f3,t0_7,X).
branch_struct(b6, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), le(f2,t0_8,X).
branch_struct(b7, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), le(f0,t0_17,X), le(f3,t0_25,X), le(f2,t0_29,X).
branch_struct(b8, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), le(f0,t0_17,X), le(f3,t0_25,X), gt(f2,t0_29,X), le(f3,t0_32,X).
branch_struct(b9, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), le(f0,t0_17,X), le(f3,t0_25,X), gt(f2,t0_29,X), gt(f3,t0_32,X).
branch_struct(b10, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), le(f0,t0_17,X), gt(f3,t0_25,X).
branch_struct(b11, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), gt(f0,t0_17,X), le(f3,t0_26,X).
branch_struct(b12, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), le(f1,t0_15,X), gt(f0,t0_17,X), gt(f3,t0_26,X).
branch_struct(b13, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), gt(f1,t0_15,X), le(f2,t0_18,X).
branch_struct(b14, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), le(f3,t0_11,X), gt(f1,t0_15,X), gt(f2,t0_18,X).
branch_struct(b15, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), le(f2,t0_10,X), gt(f3,t0_11,X).
branch_struct(b16, X) :- gt(f0,t0_0,X), le(f0,t0_2,X), gt(f2,t0_5,X), gt(f2,t0_8,X), gt(f2,t0_10,X).
branch_struct(b17, X) :- gt(f0,t0_0,X), gt(f0,t0_2,X).
branch_struct(b18, X) :- le(f3,t1_0,X), le(f0,t1_1,X), le(f3,t1_3,X), le(f1,t1_5,X), le(f2,t1_7,X), le(f0,t1_9,X).
branch_struct(b19, X) :- le(f3,t1_0,X), le(f0,t1_1,X), le(f3,t1_3,X), le(f1,t1_5,X), le(f2,t1_7,X), gt(f0,t1_9,X), le(f3,t1_12,X).
branch_struct(b20, X) :- le(f3,t1_0,X), le(f0,t1_1,X), le(f3,t1_3,X), le(f1,t1_5,X), le(f2,t1_7,X), gt(f0,t1_9,X), gt(f3,t1_12,X).
branch_struct(b21, X) :- le(f3,t1_0,X), le(f0,t1_1,X), le(f3,t1_3,X), le(f1,t1_5,X), gt(f2,t1_7,X).
branch_struct(b22, X) :- le(f3,t1_0,X), le(f0,t1_1,X), le(f3,t1_3,X), gt(f1,t1_5,X).
branch_struct(b23, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), le(f2,t1_6,X), le(f1,t1_21,X), le(f2,t1_25,X).
branch_struct(b24, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), le(f2,t1_6,X), le(f1,t1_21,X), gt(f2,t1_25,X), le(f3,t1_28,X).
branch_struct(b25, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), le(f2,t1_6,X), le(f1,t1_21,X), gt(f2,t1_25,X), gt(f3,t1_28,X).
branch_struct(b26, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), le(f2,t1_6,X), gt(f1,t1_21,X).
branch_struct(b27, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), gt(f2,t1_6,X), le(f0,t1_22,X).
branch_struct(b28, X) :- le(f3,t1_0,X), le(f0,t1_1,X), gt(f3,t1_3,X), gt(f2,t1_6,X), gt(f0,t1_22,X).
branch_struct(b29, X) :- le(f3,t1_0,X), gt(f0,t1_1,X), le(f0,t1_4,X), le(f2,t1_15,X).
branch_struct(b30, X) :- le(f3,t1_0,X), gt(f0,t1_1,X), le(f0,t1_4,X), gt(f2,t1_15,X), le(f3,t1_18,X).
branch_struct(b31, X) :- le(f3,t1_0,X), gt(f0,t1_1,X), le(f0,t1_4,X), gt(f2,t1_15,X), gt(f3,t1_18,X).
branch_struct(b32, X) :- le(f3,t1_0,X), gt(f0,t1_1,X), gt(f0,t1_4,X).
branch_struct(b33, X) :- gt(f3,t1_0,X).
branch_struct(b34, X) :- le(f0,t2_0,X), le(f0,t2_1,X).
branch_struct(b35, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), le(f3,t2_7,X).
branch_struct(b36, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), le(f2,t2_10,X), le(f3,t2_13,X), le(f2,t2_23,X).
branch_struct(b37, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), le(f2,t2_10,X), le(f3,t2_13,X), gt(f2,t2_23,X).
branch_struct(b38, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), le(f2,t2_10,X), gt(f3,t2_13,X).
branch_struct(b39, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), le(f2,t2_14,X), le(f0,t2_15,X), le(f3,t2_27,X).
branch_struct(b40, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), le(f2,t2_14,X), le(f0,t2_15,X), gt(f3,t2_27,X).
branch_struct(b41, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), le(f2,t2_14,X), gt(f0,t2_15,X).
branch_struct(b42, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), gt(f2,t2_14,X), le(f1,t2_16,X).
branch_struct(b43, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), gt(f2,t2_14,X), gt(f1,t2_16,X), le(f3,t2_18,X).
branch_struct(b44, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), gt(f2,t2_14,X), gt(f1,t2_16,X), gt(f3,t2_18,X), le(f0,t2_20,X).
branch_struct(b45, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), le(f1,t2_6,X), gt(f3,t2_7,X), gt(f2,t2_10,X), gt(f2,t2_14,X), gt(f1,t2_16,X), gt(f3,t2_18,X), gt(f0,t2_20,X).
branch_struct(b46, X) :- le(f0,t2_0,X), gt(f0,t2_1,X), gt(f1,t2_6,X).
branch_struct(b47, X) :- gt(f0,t2_0,X), le(f3,t2_2,X).
branch_struct(b48, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), le(f2,t2_4,X).
branch_struct(b49, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), gt(f2,t2_4,X), le(f3,t2_12,X), le(f3,t2_31,X), le(f1,t2_33,X).
branch_struct(b50, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), gt(f2,t2_4,X), le(f3,t2_12,X), le(f3,t2_31,X), gt(f1,t2_33,X), le(f3,t2_36,X).
branch_struct(b51, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), gt(f2,t2_4,X), le(f3,t2_12,X), le(f3,t2_31,X), gt(f1,t2_33,X), gt(f3,t2_36,X).
branch_struct(b52, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), gt(f2,t2_4,X), le(f3,t2_12,X), gt(f3,t2_31,X).
branch_struct(b53, X) :- gt(f0,t2_0,X), gt(f3,t2_2,X), gt(f2,t2_4,X), gt(f3,t2_12,X).
branch_struct(b54, X) :- le(f2,t3_0,X), le(f3,t3_1,X).
branch_struct(b55, X) :- le(f2,t3_0,X), gt(f3,t3_1,X), le(f1,t3_4,X), le(f3,t3_25,X).
branch_struct(b56, X) :- le(f2,t3_0,X), gt(f3,t3_1,X), le(f1,t3_4,X), gt(f3,t3_25,X).
branch_struct(b57, X) :- le(f2,t3_0,X), gt(f3,t3_1,X), gt(f1,t3_4,X), le(f0,t3_26,X), le(f2,t3_29,X).
branch_struct(b58, X) :- le(f2,t3_0,X), gt(f3,t3_1,X), gt(f1,t3_4,X), le(f0,t3_26,X), gt(f2,t3_29,X).
branch_struct(b59, X) :- le(f2,t3_0,X), gt(f3,t3_1,X), gt(f1,t3_4,X), gt(f0,t3_26,X).
branch_struct(b60, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), le(f1,t3_7,X), le(f1,t3_15,X).
branch_struct(b61, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), le(f1,t3_7,X), gt(f1,t3_15,X), le(f2,t3_18,X), le(f2,t3_19,X).
branch_struct(b62, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), le(f1,t3_7,X), gt(f1,t3_15,X), le(f2,t3_18,X), gt(f2,t3_19,X), le(f0,t3_22,X).
branch_struct(b63, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), le(f1,t3_7,X), gt(f1,t3_15,X), le(f2,t3_18,X), gt(f2,t3_19,X), gt(f0,t3_22,X).
branch_struct(b64, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), le(f1,t3_7,X), gt(f1,t3_15,X), gt(f2,t3_18,X).
branch_struct(b65, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), le(f0,t3_5,X), gt(f1,t3_7,X).
branch_struct(b66, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), gt(f0,t3_5,X), le(f3,t3_8,X).
branch_struct(b67, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), gt(f0,t3_5,X), gt(f3,t3_8,X), le(f3,t3_10,X), le(f2,t3_11,X).
branch_struct(b68, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), gt(f0,t3_5,X), gt(f3,t3_8,X), le(f3,t3_10,X), gt(f2,t3_11,X).
branch_struct(b69, X) :- gt(f2,t3_0,X), le(f0,t3_2,X), gt(f0,t3_5,X), gt(f3,t3_8,X), gt(f3,t3_10,X).
branch_struct(b70, X) :- gt(f2,t3_0,X), gt(f0,t3_2,X).
branch_struct(b71, X) :- le(f3,t4_0,X).
branch_struct(b72, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), le(f2,t4_3,X), le(f0,t4_5,X), le(f1,t4_15,X).
branch_struct(b73, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), le(f2,t4_3,X), le(f0,t4_5,X), gt(f1,t4_15,X), le(f2,t4_18,X).
branch_struct(b74, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), le(f2,t4_3,X), le(f0,t4_5,X), gt(f1,t4_15,X), gt(f2,t4_18,X).
branch_struct(b75, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), le(f2,t4_3,X), gt(f0,t4_5,X).
branch_struct(b76, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), le(f2,t4_6,X), le(f3,t4_7,X).
branch_struct(b77, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), le(f2,t4_6,X), gt(f3,t4_7,X), le(f2,t4_40,X), le(f3,t4_41,X).
branch_struct(b78, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), le(f2,t4_6,X), gt(f3,t4_7,X), le(f2,t4_40,X), gt(f3,t4_41,X).
branch_struct(b79, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), le(f2,t4_6,X), gt(f3,t4_7,X), gt(f2,t4_40,X).
branch_struct(b80, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), gt(f2,t4_6,X), le(f0,t4_8,X), le(f0,t4_27,X).
branch_struct(b81, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), gt(f2,t4_6,X), le(f0,t4_8,X), gt(f0,t4_27,X), le(f3,t4_30,X).
branch_struct(b82, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), gt(f2,t4_6,X), le(f0,t4_8,X), gt(f0,t4_27,X), gt(f3,t4_30,X).
branch_struct(b83, X) :- gt(f3,t4_0,X), le(f0,t4_2,X), gt(f2,t4_3,X), gt(f2,t4_6,X), gt(f0,t4_8,X).
branch_struct(b84, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), le(f3,t4_9,X), le(f2,t4_11,X), le(f1,t4_13,X), le(f2,t4_33,X).
branch_struct(b85, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), le(f3,t4_9,X), le(f2,t4_11,X), le(f1,t4_13,X), gt(f2,t4_33,X), le(f2,t4_36,X).
branch_struct(b86, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), le(f3,t4_9,X), le(f2,t4_11,X), le(f1,t4_13,X), gt(f2,t4_33,X), gt(f2,t4_36,X).
branch_struct(b87, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), le(f3,t4_9,X), le(f2,t4_11,X), gt(f1,t4_13,X).
branch_struct(b88, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), le(f3,t4_9,X), gt(f2,t4_11,X).
branch_struct(b89, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), le(f0,t4_4,X), gt(f3,t4_9,X).
branch_struct(b90, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), gt(f0,t4_4,X), le(f0,t4_10,X), le(f1,t4_21,X).
branch_struct(b91, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), gt(f0,t4_4,X), le(f0,t4_10,X), gt(f1,t4_21,X), le(f1,t4_24,X).
branch_struct(b92, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), gt(f0,t4_4,X), le(f0,t4_10,X), gt(f1,t4_21,X), gt(f1,t4_24,X).
branch_struct(b93, X) :- gt(f3,t4_0,X), gt(f0,t4_2,X), gt(f0,t4_4,X), gt(f0,t4_10,X).
branch_struct(b94, X) :- le(f3,t5_0,X), le(f3,t5_1,X).
branch_struct(b95, X) :- le(f3,t5_0,X), gt(f3,t5_1,X).
branch_struct(b96, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), le(f2,t5_7,X), le(f3,t5_9,X).
branch_struct(b97, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), le(f2,t5_7,X), gt(f3,t5_9,X), le(f2,t5_14,X).
branch_struct(b98, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), le(f2,t5_7,X), gt(f3,t5_9,X), gt(f2,t5_14,X).
branch_struct(b99, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), gt(f2,t5_7,X), le(f3,t5_10,X), le(f1,t5_11,X), le(f1,t5_17,X).
branch_struct(b100, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), gt(f2,t5_7,X), le(f3,t5_10,X), le(f1,t5_11,X), gt(f1,t5_17,X).
branch_struct(b101, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), gt(f2,t5_7,X), le(f3,t5_10,X), gt(f1,t5_11,X).
branch_struct(b102, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), le(f3,t5_5,X), gt(f2,t5_7,X), gt(f3,t5_10,X).
branch_struct(b103, X) :- gt(f3,t5_0,X), le(f2,t5_2,X), gt(f3,t5_5,X).
branch_struct(b104, X) :- gt(f3,t5_0,X), gt(f2,t5_2,X).
branch_struct(b105, X) :- le(f2,t6_0,X), le(f2,t6_1,X), le(f2,t6_3,X).
branch_struct(b106, X) :- le(f2,t6_0,X), le(f2,t6_1,X), gt(f2,t6_3,X).
branch_struct(b107, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), le(f2,t6_4,X), le(f3,t6_7,X).
branch_struct(b108, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), le(f2,t6_4,X), gt(f3,t6_7,X), le(f0,t6_16,X).
branch_struct(b109, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), le(f2,t6_4,X), gt(f3,t6_7,X), gt(f0,t6_16,X).
branch_struct(b110, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), gt(f2,t6_4,X), le(f3,t6_8,X), le(f2,t6_9,X).
branch_struct(b111, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), gt(f2,t6_4,X), le(f3,t6_8,X), gt(f2,t6_9,X).
branch_struct(b112, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), gt(f2,t6_4,X), gt(f3,t6_8,X), le(f3,t6_10,X).
branch_struct(b113, X) :- le(f2,t6_0,X), gt(f2,t6_1,X), gt(f2,t6_4,X), gt(f3,t6_8,X), gt(f3,t6_10,X).
branch_struct(b114, X) :- gt(f2,t6_0,X).
branch_struct(b115, X) :- le(f3,t7_0,X), le(f3,t7_1,X).
branch_struct(b116, X) :- le(f3,t7_0,X), gt(f3,t7_1,X).
branch_struct(b117, X) :- gt(f3,t7_0,X), le(f2,t7_2,X), le(f3,t7_3,X).
branch_struct(b118, X) :- gt(f3,t7_0,X), le(f2,t7_2,X), gt(f3,t7_3,X), le(f0,t7_22,X).
branch_struct(b119, X) :- gt(f3,t7_0,X), le(f2,t7_2,X), gt(f3,t7_3,X), gt(f0,t7_22,X).
branch_struct(b120, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), le(f2,t7_4,X).
branch_struct(b121, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), le(f1,t7_9,X).
branch_struct(b122, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), gt(f1,t7_9,X), le(f0,t7_12,X), le(f3,t7_13,X).
branch_struct(b123, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), gt(f1,t7_9,X), le(f0,t7_12,X), gt(f3,t7_13,X).
branch_struct(b124, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), gt(f1,t7_9,X), gt(f0,t7_12,X), le(f1,t7_14,X).
branch_struct(b125, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), gt(f1,t7_9,X), gt(f0,t7_12,X), gt(f1,t7_14,X), le(f3,t7_18,X).
branch_struct(b126, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), le(f3,t7_8,X), gt(f1,t7_9,X), gt(f0,t7_12,X), gt(f1,t7_14,X), gt(f3,t7_18,X).
branch_struct(b127, X) :- gt(f3,t7_0,X), gt(f2,t7_2,X), gt(f2,t7_4,X), gt(f3,t7_8,X).