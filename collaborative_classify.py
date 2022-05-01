import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime


import mysql.connector
from mysql.connector import Error
import copy as copy

import altair as alt

import streamlit as st




sqlserver = st.secrets["sqlserver"]
sqlport = st.secrets["sqlport"]
sqluser = st.secrets["sqluser"]
sqlpwd = st.secrets["sqlpwd"]


def create_connection(host_name, port_no, user_name, user_password,dbname):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            port=port_no,
            user=user_name,
            passwd=user_password,
            database=dbname
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def df_to_sql_insert(df_in,indexer=True,startval=0):
    indval = 1*startval
    data_for_query = ""
    for j in range(len(df_in)):
        curline = ""
        for element in ['"'*int(not str(element).replace('.','',1).isnumeric()) + str(element) + '"'*int(not str(element).replace('.','',1).isnumeric())  + "," for element in np.array(df_in.loc[df_in.index[j]])]:
            curline += element
        data_for_query += "(" 
        if indexer:
            data_for_query += str(indval) + ","
            indval += 1
        data_for_query += curline[0:-1] + ")," + "\n"
    return data_for_query[0:-2]

def get_current_max_ind(connect,table,indname,execute_read_query=execute_read_query):
    query = "SELECT MAX(" + indname + ") FROM " + table + ";"
    curmaxind = execute_read_query(connection,query)[0][0]
    if curmaxind is None:
        curmaxind = -1
    return curmaxind

def integer_onetcode(thedf,onetcolumn):
    return np.array(thedf[onetcolumn].str[:2].astype(np.int64)*1000000 \
                 + thedf[onetcolumn].str[3:7].astype(np.int64)*100 \
                 + thedf[onetcolumn].str[-2:].astype(np.int64),dtype=np.int64)


### Construct sensitivity and specificity vectors, with companion vector of classification cut-offs
def sens_spec_vec(probs,data,Ncutoffs=101):
    realpos = np.sum(data == 1)
    realneg = np.sum(data == 0)

    cutoffvec = np.linspace(0,1,Ncutoffs)
    
    sens = np.array([np.sum((probs.flatten() >= x) & (data.flatten() == 1))/realpos for x in cutoffvec])
    spec = np.array([np.sum((probs.flatten() < x) & (data.flatten() == 0))/realneg for x in cutoffvec])
    prec = np.array([np.sum((probs.flatten() >= x) & (data.flatten() == 1))/np.sum(probs.flatten() >= x) for x in cutoffvec])
    idpos = np.array([np.sum(probs.flatten() >= x)    for x in cutoffvec])
    idneg = np.array([np.sum(probs.flatten() < x)    for x in cutoffvec])
    truepos = np.array([np.sum((probs.flatten() >= x) & (data.flatten() == 1))    for x in cutoffvec])
    trueneg = np.array([np.sum((probs.flatten() < x) & (data.flatten() == 0))    for x in cutoffvec])
    return sens,spec,cutoffvec,prec,realpos,realneg,idpos,idneg,truepos,trueneg

### calculate log odds ratios and probabilities given Logit parameters and data
def logit_probs(X,bets):    
    logodds = np.dot(X,bets) # calculate log odds ratios
    probs = 1/(1+np.exp(-logodds)) # calculate Logit probabilities
    
    return logodds,probs

### calculate a variety of fit statistics for Logit regression results
def logit_fit(X,y,bets,sens_spec_vec=sens_spec_vec,logit_probs=logit_probs):
    N = len(y)
    logodds,probs = logit_probs(X,bets)
    
    if any(np.isnan(probs)): # check for any catastrophic numerical failures. Not very likely, but can't hurt to be prepared.
        LLn = -np.Inf
    else:
        # calculate the elements of the sum for which data = 1
        likelyvec1 = y.flatten()*np.log(probs.flatten())                   # for extreme values, this can generate numerical errors
        likelyvec1[(y.flatten() == 0) & (probs.flatten() == 0)] = 0        # if data = 0, contribution to this part of the sum is zero
        likelyvec1[(y.flatten() == 1) & (probs.flatten() == 0)] = -np.Inf  # if data = 1 and estimated probability is zero, assign -Inf
        
        # calculate the elements of the sum for which data = 0
        likelyvec2 = (1-y.flatten())*np.log(1-probs.flatten())             # for extreme values, this can generate numerical errors
        likelyvec2[(y.flatten() == 1) & (probs.flatten() == 1)] = 0        # if data = 1, contribution to this part of the sum is zero
        likelyvec2[(y.flatten() == 0) & (probs.flatten() == 1)] = -np.Inf  # if data = 0 and estimated probability is 1, assign -Inf
        
        # sum it all up
        likelyvec = likelyvec1 + likelyvec2
        LLn = np.sum(likelyvec)
    
    ybar = np.mean(y)
    LLnybar = N*(ybar*np.log(ybar) + (1 - ybar)*np.log(1 - ybar)  )
    pseudoR2 = 1 - LLn/LLnybar
    
    sens_local,spec_local,cutoffs,prec,realpos,realneg,idpos,idneg,truepos,trueneg = sens_spec_vec(probs,y)
    AUC = np.sum((spec_local[1:]-spec_local[0:-1])*(sens_local[1:] + sens_local[0:-1])/2)
    
    return logodds,probs,LLn,pseudoR2,AUC,sens_local,spec_local,cutoffs,prec,realpos,realneg,idpos,idneg,truepos,trueneg


### logit-estimating meta-function with multiple layers of fail-safes
### developed with Statsmodels standard logit function, but might be adapted to some other
def robust_logitter(X,y,disp=0,maxiter=10000,logitfunc = sm.Logit):
    
    logit_mod = sm.Logit(y,X) # initialize the model

    try:
        logit_res = logit_mod.fit(maxiter=maxiter,disp=disp) # try straight-up, with the standard derivative-based optimizer
    except:
        try:
            # if that fails, try with Nelder-Mead, which is derivative-free
            logit_res = logit_mod.fit(maxiter=maxiter,method='nm',disp=disp) 
            try:
                # then, feed the optimized parameters back in as a starting point and try the standard algorithm again
                logit_res = logit_mod.fit(maxiter=maxiter,start_params=logit_res.params,disp=disp) 
            except:
                # if that does not work just use a regularized objective function
                # if the meta-algorithm reaches this point, it's usually due to perfect separation
                logit_res = logit_mod.fit_regularized(maxiter=maxiter,alpha=.01,disp=disp)
        except:
            # if even Nelder-Mead fails because of perfect separation, use a regularized objective function
            logit_res = logit_mod.fit_regularized(maxiter=maxiter,alpha=.01,disp=disp)
    return logit_res

@st.cache(allow_output_mutation=True,ttl=3600)
def cached_logit(datain,includecomp,TTt):
    yy = np.reshape(np.array(datain['user_classify']),(len(datain),1))

    Xz = np.ones((len(yy),0))
    
    Xz = np.hstack((Xz,np.array(datain[["PC" + str(x) for x in np.arange(1,10)[includecomp]]])))
    
    subidlist = []
    subidcount = []
    for subid in pd.unique(datain["submission_id"]):
        subidlist.append(subid)
        subidcount.append(np.sum(np.array(datain["submission_id"]) == subid))
        Xz = np.hstack((Xz,1.*np.reshape(np.array(datain["submission_id"]) == subid,(len(yy),1))))
    
    logit_results = robust_logitter(Xz,yy,disp=0)
    logit_yhat,yy_hat,LLn,pseudoR2,AUC,sens,spec,cutoffs,prec,realpos,realneg,idpos,idneg,truepos,trueneg=logit_fit(Xz,yy,logit_results.params)
    average_const_coeff = np.sum(np.array(subidcount)*np.array(logit_results.params[-len(subidcount):]))\
                        /np.sum(np.array(subidcount))
    average_params = np.array(list(logit_results.params[0:-len(subidcount)]) + [average_const_coeff])

    Xz_complete = np.hstack((TTt[:,[x for x in np.arange(9)[includecomp]]],np.ones((len(TTt[:,0]),1))))
    _,yhat = logit_probs(Xz_complete,average_params)
    return logit_results,average_params,yhat,AUC,sens,spec,cutoffs,prec,realpos,realneg,idpos,idneg,truepos,trueneg


@st.cache
def data_load_and_PCA(filenmain,fileinvarinf):
    df_data = pd.read_parquet(filenmain)
    df_varinfo = pd.read_parquet(fileinvarinf)
    XX = np.array(df_data.drop(columns=["O*NET-SOC Code","Title","Description"]))
    n,k = np.shape(XX)
    xx = XX - np.reshape(np.sum(XX,axis=0),(1,k))/(n)
    vv,WW = np.linalg.eig(np.dot(xx.T,xx))
    TT = np.dot(xx,WW)
    return df_data,df_varinfo,XX,n,k,xx,vv,WW,TT



@st.cache(ttl=3600)
def load_from_sql(sqlquery,sqlcreds,columnnames,execute_read_query=execute_read_query,create_connection=create_connection):
    sqlconnection = create_connection(*sqlcreds)
    datainput = execute_read_query(sqlconnection,sqlquery)
    sqlconnection.close()
    return pd.DataFrame(data=datainput,columns=columnnames)

@st.cache(ttl=3600)
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache(ttl=3600)
def build_dataset(ttin,kk,occupin,classdatin,theattr):
    pcomp = pd.DataFrame(data=ttin,columns=["PC" + str(j+1) for j in range(kk)])
    pcomp["O*NET-SOC Code"] = np.array(occupin["O*NET-SOC Code"])

    return classdatin.loc[np.array(df_classdata["attribute"]) == theattr]\
                .merge(pcomp,how="left",left_on="ONET_string",right_on="O*NET-SOC Code")

query = """
SELECT 
    cl.entry_id,
    cl.submission_id,
    su.user_name,
    su.add_time,
    cl.occup_id,
    oc.ONET_occup,
    oc.occup_name,
    cl.attribute,
    cl.user_classify 
FROM 
    classify as cl
INNER JOIN
    occupations as oc
ON
    cl.occup_id = oc.occup_id
INNER JOIN
    submissions as su
ON
    su.submission_id = cl.submission_id;
"""

colnames = ["entry_id","submission_id","user_name","add_time","ONET_int","ONET_string","occup_name","attribute","user_classify"]

df_classdata = load_from_sql(query,(sqlserver,sqlport,sqluser,sqlpwd,'occu_classify'),colnames)


query = """
SELECT 
    oc.occup_id,
    MAX(oc.ONET_occup),
    MAX(oc.occup_name),
    cl.attribute,
    COUNT(CASE cl.user_classify WHEN True THEN 1 ELSE NULL END),
    COUNT(CASE cl.user_classify WHEN False THEN 1 ELSE NULL END)
FROM 
    occupations as oc
LEFT OUTER JOIN
    classify as cl
ON
    cl.occup_id = oc.occup_id
GROUP BY
    oc.occup_id,
    cl.attribute;
"""

colnames = ["ONET_int","ONET_string","occup_name","attribute","user_pos","user_neg"]

df_classummary_byoccup = load_from_sql(query,(sqlserver,sqlport,sqluser,sqlpwd,'occu_classify'),colnames)


query = """
SELECT 
    cl.submission_id,
    MAX(su.add_time),
    DATE(MAX(su.add_time)),
    MAX(cl.attribute), 
    MAX(su.user_name), 
    COUNT(DISTINCT cl.entry_id),
    CAST(SUM(COUNT(DISTINCT cl.entry_id)) OVER(PARTITION BY MAX(cl.attribute) ORDER BY MAX(su.add_time) ) AS UNSIGNED)  
FROM 
    classify as cl
INNER JOIN
    submissions as su
ON
    su.submission_id = cl.submission_id
GROUP BY
    cl.submission_id;
"""

colnames = ["submission_id","add_time","add_date","attribute","user_name","entry_count","cumulative_count"]

df_submithistory = load_from_sql(query,(sqlserver,sqlport,sqluser,sqlpwd,'occu_classify'),colnames)



query = """
SELECT 
    cl.attribute,
    COUNT(DISTINCT cl.submission_id),
    COUNT(DISTINCT su.user_name), 
    COUNT(DISTINCT cl.entry_id)
FROM 
    classify as cl
INNER JOIN
    submissions as su
ON
    su.submission_id = cl.submission_id
GROUP BY
    cl.attribute;
"""

colnames = ["attribute","submission_count","contributor_count","entry_count"]

df_sm_byattr = load_from_sql(query,(sqlserver,sqlport,sqluser,sqlpwd,'occu_classify'),colnames)


df_occupdata,df_occupvarinfo,XX,n,k,xx,vv,WW,TT = data_load_and_PCA("occup_data_toanalyze.gzip","occup_varnames.gzip")

@st.cache(ttl=3600)
def chartify_submithistory(df_in):
    df_out = df_in.copy()
    df_out["date_to_chart"] = df_out["add_date"].astype('datetime64') + datetime.timedelta(hours=7)
    for attr in list(pd.unique(df_out["attribute"])):
        df_out = df_out.append(pd.DataFrame({
                                      "add_time":[df_out["add_time"].min(),datetime.datetime.utcnow()],
                                      "attribute":[attr,attr],
                                      "cumulative_count":[df_out.loc[df_in["attribute"] == attr,"cumulative_count"].min(),
                                                          df_out.loc[df_in["attribute"] == attr,"cumulative_count"].max()]}),
                                    )
    return df_out

#df_submithistory_tochart["date_to_chart"] = df_submithistory_tochart["add_date"].astype('datetime64') + datetime.timedelta(hours=7)
#for attr in list(pd.unique(df_submithistory["attribute"])):
#    df_submithistory_tochart = df_submithistory_tochart.append(pd.DataFrame({
#                                  "add_time":[df_submithistory["add_time"].min(),datetime.datetime.utcnow()],
#                                  "attribute":[attr,attr],
#                                  "cumulative_count":[df_submithistory.loc[df_submithistory["attribute"] == attr,"cumulative_count"].min(),
#                                                      df_submithistory.loc[df_submithistory["attribute"] == attr,"cumulative_count"].max()]}),
#                                )

df_submithistory_tochart = chartify_submithistory(df_submithistory)


contrib_chart = alt.Chart(df_submithistory_tochart,title="Contributions over time")

highlighter = alt.selection_single(on='mouseover')
chartwidth = (df_submithistory["add_date"].max().day-df_submithistory["add_date"].min().day)
min_add_time = df_submithistory["add_time"].min()
max_add_time = df_submithistory["add_time"].max()

@st.cache(ttl=3600)
def build_summary_charts(basechart,widthparam,minmaxtimes,hlelement):
    out_bar = basechart.mark_bar(width=30*widthparam/2).encode(x=alt.X("date_to_chart",
                                                                  axis=alt.Axis(format="%B %d, %Y",
                                                                                labelAngle=-45,
                                                                                tickOffset=0,
                                                                                tickMinStep=86400000,
                                                                                grid=False
                                                                                ),
                                                                  scale=alt.Scale(domain=[minmaxtimes[0].timestamp()*1000-86400000/4,
                                                                                          minmaxtimes[1].timestamp()*1000+86400000/4],clamp=True,nice=False)
                                                                  ),
                                                  y="entry_count",
                                                  color="attribute",
                                                  opacity=alt.condition(~hlelement,alt.value(0.9),alt.value(0.5)),
                                                  tooltip=[alt.Tooltip("add_time",title="When",format="%B %d, %Y"),
                                                           alt.Tooltip("entry_count",title="No. of entries"),
                                                           alt.Tooltip("user_name",title="Who")]).add_selection(hlelement)

    out_cumu = basechart.mark_line(opacity=.8).encode(x=alt.X("add_time",
                                                                  axis=alt.Axis(format="%B %d, %Y",
                                                                                labelAngle=-45,
                                                                                tickOffset=0,
                                                                                tickMinStep=86400000,
                                                                                grid=False
                                                                                ),
                                                                  scale=alt.Scale(domain=[minmaxtimes[0].timestamp()*1000-86400000/4,
                                                                                          minmaxtimes[1].timestamp()*1000+86400000/4],clamp=True,nice=False)
                                                                  ),
                                                    y="cumulative_count",
                                                    color="attribute",
                                                    tooltip=[alt.Tooltip("add_time",title="When",format="%B %d, %Y"),
                                                             alt.Tooltip("cumulative_count",title="No. of entries")])
    return out_bar,out_cumu


contrib_bar, contrib_cumu = build_summary_charts(contrib_chart,chartwidth,[min_add_time,max_add_time],highlighter)
   
#contrib_bar = contrib_chart.mark_bar(width=30*chartwidth/2).encode(x=alt.X("date_to_chart",
#                                                              axis=alt.Axis(format="%B %d, %Y",
#                                                                            labelAngle=-45,
#                                                                            tickOffset=0,
#                                                                            tickMinStep=86400000,
#                                                                            grid=False
#                                                                            ),
#                                                              scale=alt.Scale(domain=[df_submithistory["add_time"].min().timestamp()*1000-86400000/4,
#                                                                                      df_submithistory["add_time"].max().timestamp()*1000+86400000/4],clamp=True,nice=False)
#                                                              ),
#                                              y="entry_count",
#                                              color="attribute",
#                                              opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
#                                              tooltip=[alt.Tooltip("add_time",title="When",format="%B %d, %Y"),
#                                                       alt.Tooltip("entry_count",title="No. of entries"),
#                                                       alt.Tooltip("user_name",title="Who")]).add_selection(highlighter)
#
#contrib_cumu = contrib_chart.mark_line(opacity=.8).encode(x=alt.X("add_time",
#                                                              axis=alt.Axis(format="%B %d, %Y",
#                                                                            labelAngle=-45,
#                                                                            tickOffset=0,#-75+30*chartwidth2/212122000,
#                                                                            tickMinStep=86400000,
#                                                                            grid=False
#                                                                            ),
#                                                              scale=alt.Scale(domain=[df_submithistory["add_time"].min().timestamp()*1000-86400000/4,
#                                                                                      df_submithistory["add_time"].max().timestamp()*1000+86400000/4],clamp=True,nice=False)
#                                                              ),
#                                                y="cumulative_count",
#                                                color="attribute",
#                                                tooltip=[alt.Tooltip("add_time",title="When",format="%B %d, %Y"),
#                                                         alt.Tooltip("cumulative_count",title="No. of entries")])


summarytable = f"""##### Human-assigned classifications by attribute
| Attribute | No. of contributions | No. of contributors | Total entry count |
| ---       | ---                  | ---                 | ---               |"""

for j in range(len(df_sm_byattr)):
    summarytable += "\n" + f"""| {df_sm_byattr["attribute"][j]} | {df_sm_byattr["submission_count"][j]}"""
    summarytable += f""" | {df_sm_byattr["contributor_count"][j]} | {df_sm_byattr["entry_count"][j]} |"""



st.title("Collaborative Classify: O*NET Occupations")
st.write("""
This tool pools training sets contributed by users of the [Occupation classifier](https://mattdelventhal.com/project/occu_classify/ "Occupation classifier") and proceeds to a deeper
analysis of occupation attributes for 873 Department of Labor [O\*NET occupations](https://www.onetonline.org/ "O*NET Online website"). 
Larger training sets allow the machine learning algorithm to base its predictions on a wider set of variables
with less risk of over-fitting.""")


overview = st.expander("Overview")


overview.write("""
The **Summary of data contributions** tab presents information on contributions by users of the Occupation classifier
app: How many manual classifications were submitted, for which attributes, when and by whom.

The **Classify one attribute** tab leverages these contributions into an in-depth analysis of one attribute. You may select which principal
components of the underlying data to use for prediction, and set a cut-off threshold for classification. You can watch as the results of the
machine learning algorithm, and basic statistics like *Sensitivity*, *Specificity*, *Precision* and *Recall* are updated in real time.

The **Compare two attributes** plots classification results for two distinct attributes against each other. This allows you to see, for 
example, if there are any occupations which are both "Prestigious" *and* "Dangerous."

The **Methodology** tab provides information on the machine learning algorithm used and O/*NET occupation data which it uses to draw inferences
from human attribute classifications.

If you like this app, please head over to the [Occupation classifier](https://mattdelventhal.com/project/occu_classify/ "Occupation classifier") and contribute a few classifications of your own!
"""
)

firstdiv = st.expander("Summary of data contributions")
nextdiv = st.expander("Classify one attribute")
finaldiv = st.expander("Compare two attributes")
methodology = st.expander("Methodology + data download")

with firstdiv:
    tablealigncols1 = st.columns([1,3,1])
    tablealigncols1[1].write(summarytable,unsafe_allow_html=True)

    st.write(" ")
    st.write(" ")

    charcols1 = st.columns([1,6])

    contrib_view_type = charcols1[0].radio("View type:",["raw","cumulative"])
    if contrib_view_type == "raw":
        charcols1[1].altair_chart(contrib_bar.properties(width=600),use_container_width=True)
    elif contrib_view_type == "cumulative":
        charcols1[1].altair_chart(contrib_cumu.properties(width=600),use_container_width=True)


with nextdiv:
    attribute = st.selectbox("Select an attribute to analyze:",list(pd.unique(df_classdata["attribute"].sort_values())))

    st.write("###### Which principal components to include:")
    PC_cb_cols = st.columns(10)
    st.write(" ")



    ordinalnums = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th"]
    defaults = [True,True,True,True,True,False,False,False,False]
    includePC = [False]*9
    for j in range(9):
        includePC[j] = PC_cb_cols[j+1].checkbox(ordinalnums[j],value=defaults[j])



    


    df_curdata = build_dataset(TT,k,df_occupdata,df_classdata,attribute)
    #df_pcomp = pd.DataFrame(data=TT,columns=["PC" + str(j+1) for j in range(k)])
    #df_pcomp["O*NET-SOC Code"] = np.array(df_occupdata["O*NET-SOC Code"])
    
    #df_curdata = df_classdata.loc[np.array(df_classdata["attribute"]) == attribute].merge(df_pcomp,how="left",left_on="ONET_string",right_on="O*NET-SOC Code")



    logit_res,average_params,yhat,AUC,sens,spec,cutoffs,prec,realpos,realneg,idpos,idneg,truepos,trueneg = cached_logit(df_curdata,includePC,TT)
    
        
    logit_resc = copy.copy(logit_res)

    logit_result_cols = st.columns([8,3])
    
    logit_result_cols[1].write(f"""
    |   |     |
    | --- | --- |
     **N. Obs.**   | {len(df_curdata)}
     **Pseudo-R$^{2}$**   |  {logit_resc.prsquared:.2f}
     **AUC**  |  {AUC:.2f}
    """)
    @st.cache(ttl=3600)    
    def make_coeff_chart(paramsin,inclpcin,bsein,hlelement):
        coeffs_df = pd.DataFrame({"coeff_value":paramsin[0:np.sum(np.array(inclpcin))],
                                  "se":bsein[0:np.sum(np.array(inclpcin))],
                                  "PC_number":np.arange(1,10)[inclpcin],
                                  "lowerbound":np.array(paramsin[0:np.sum(np.array(inclpcin))]) - 1.96*np.array(bsein[0:np.sum(np.array(inclpcin))]),
                                  "upperbound":np.array(paramsin[0:np.sum(np.array(inclpcin))]) + 1.96*np.array(bsein[0:np.sum(np.array(inclpcin))])
                                })
        coeff_chart = alt.Chart(coeffs_df,width=400,height=200)
        squaresout = coeff_chart.mark_square().encode(x=alt.X("PC_number",axis=alt.Axis(grid=False,title="Principal Component"),scale=alt.Scale(domain=[0.5,9.5],clamp=True,nice=False)),
                                                   y=alt.Y("coeff_value",axis=alt.Axis(grid=False,title="Estimated Coefficient")),
                                                   tooltip=[alt.Tooltip("PC_number",title="Principal Component"),
                                                            alt.Tooltip("coeff_value",title="Estimated Coeff.",format='.2f'),
                                                            alt.Tooltip("se",title="Standard Error",format='.2f')],
                                                    size=alt.condition(~hlelement,alt.value(50),alt.value(100))).add_selection(hlelement)
        errorbarsout = coeff_chart.mark_rule(color='red').encode(alt.X("PC_number"),alt.Y("lowerbound"),alt.Y2("upperbound"))
        zerolineout = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='black',strokeDash=[5,3]).encode( y='y') 
        return squaresout,errorbarsout,zerolineout


    squares,errorbars,zeroline = make_coeff_chart(logit_resc.params,includePC,logit_resc.bse,highlighter)
    
    #coeffs_df = pd.DataFrame({"coeff_value":logit_resc.params[0:np.sum(np.array(includePC))],
    #                          "se":logit_resc.bse[0:np.sum(np.array(includePC))],
    #                          "PC_number":np.arange(1,10)[includePC],
    #                          "lowerbound":np.array(logit_resc.params[0:np.sum(np.array(includePC))]) - 1.96*np.array(logit_resc.bse[0:np.sum(np.array(includePC))]),
    #                          "upperbound":np.array(logit_resc.params[0:np.sum(np.array(includePC))]) + 1.96*np.array(logit_resc.bse[0:np.sum(np.array(includePC))])
    #                        })
    #coeff_chart = alt.Chart(coeffs_df,width=480,height=240)
    #squares = coeff_chart.mark_square().encode(x=alt.X("PC_number",axis=alt.Axis(grid=False,title="Principal Component"),scale=alt.Scale(domain=[0.5,9.5],clamp=True,nice=False)),
    #                                           y=alt.Y("coeff_value",axis=alt.Axis(grid=False,title="Estimated Coefficient")),
    #                                           tooltip=[alt.Tooltip("PC_number",title="Principal Component"),
    #                                                    alt.Tooltip("coeff_value",title="Estimated Coeff.",format='.2f'),
    #                                                    alt.Tooltip("se",title="Standard Error",format='.2f')],
    #                                            size=alt.condition(~highlighter,alt.value(50),alt.value(100))).add_selection(highlighter)
    #errorbars = coeff_chart.mark_rule(color='red').encode(alt.X("PC_number"),alt.Y("lowerbound"),alt.Y2("upperbound"))
    #zeroline = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='black',strokeDash=[5,3]).encode( y='y') 

    logit_result_cols[0].altair_chart(squares + errorbars + zeroline)
    #blankchart = alt.Chart(pd.DataFrame({'xx':[1,1],'yy':[1,1]}),height=160,width=160).mark_point(opacity=0).encode(
    #            x=alt.X('xx',axis=alt.Axis(grid=False,title=None,ticks=False,labels=False,aria=False)),
    #            y=alt.Y('xx',axis=alt.Axis(grid=False,title=None,ticks=False,labels=False,aria=False)))



    #bse
    #prsquared
    @st.cache(ttl=3600)
    def augment_occupdata(occupin,classsumin,yhatter,theattr):
        df_out = occupin.copy()
        df_out = df_out[["O*NET-SOC Code","Title","Description"]]\
                            .merge(classsumin.loc[(classsumin["attribute"]==theattr) \
                                                                    | (classsumin["attribute"].isna()),
                                                               ["ONET_string","user_pos","user_neg"]],
                                        how="left",left_on="O*NET-SOC Code",right_on="ONET_string")
        df_out = df_out.drop(columns=["ONET_string"])    
        df_out["user_pos"] = df_out["user_pos"].fillna(0)
        df_out["user_neg"] = df_out["user_neg"].fillna(0)
        df_out["user_pos"] = df_out["user_pos"].astype(int)
        df_out["user_neg"] = df_out["user_neg"].astype(int)
        df_out["classified_as"] = df_out["user_pos"] - df_out["user_neg"]
        
        df_out.loc[df_out["classified_as"] < -1,"classified_as"] = -1
        df_out.loc[df_out["classified_as"] > 1,"classified_as"] = 1
        df_out[attribute + "?"] = "Not classified"
        df_out.loc[df_out["classified_as"] == -1,attribute + "?"] = "NO"
        df_out.loc[df_out["classified_as"] == 1,attribute + "?"] = "YES"
        df_out[attribute + "_score"] = yhatter.flatten()
        return df_out

    df_occupdata_a = augment_occupdata(df_occupdata,df_classummary_byoccup,yhat,attribute)
    
    #df_occupdata_a = df_occupdata.copy()
    #df_occupdata_a = df_occupdata_a[["O*NET-SOC Code","Title","Description"]]\
    #                    .merge(df_classummary_byoccup.loc[(df_classummary_byoccup["attribute"]=="Analytical") \
    #                                                            | (df_classummary_byoccup["attribute"].isna()),
    #                                                       ["ONET_string","user_pos","user_neg"]],
    #                                how="left",left_on="O*NET-SOC Code",right_on="ONET_string")
    #df_occupdata_a = df_occupdata_a.drop(columns=["ONET_string"])    
    #df_occupdata_a["user_pos"] = df_occupdata_a["user_pos"].fillna(0)
    #df_occupdata_a["user_neg"] = df_occupdata_a["user_neg"].fillna(0)
    #df_occupdata_a["user_pos"] = df_occupdata_a["user_pos"].astype(int)
    #df_occupdata_a["user_neg"] = df_occupdata_a["user_neg"].astype(int)
    #df_occupdata_a["classified_as"] = df_occupdata_a["user_pos"] - df_occupdata_a["user_neg"]
    #
    #df_occupdata_a.loc[df_occupdata_a["classified_as"] < -1,"classified_as"] = -1
    #df_occupdata_a.loc[df_occupdata_a["classified_as"] > 1,"classified_as"] = 1
    #df_occupdata_a[attribute + "?"] = "Not classified"
    #df_occupdata_a.loc[df_occupdata_a["classified_as"] == -1,attribute + "?"] = "NO"
    #df_occupdata_a.loc[df_occupdata_a["classified_as"] == 1,attribute + "?"] = "YES"
    #df_occupdata_a[attribute + "_score"] = yhat.flatten()
    
    sorted_scores = np.argsort(yhat)[::-1]
    df_sorted_occupations = pd.DataFrame({attribute + "_score":yhat.flatten()[sorted_scores],
                                          attribute + "_rank":[x + 1 for x in list(range(n))],
                                          "occup_title":list(
                                                             np.array(
                                                                    df_occupdata_a['Title']
                                                                    )[sorted_scores]
                                                                 ),
                                          attribute + "?":list(
                                                             np.array(df_occupdata_a[attribute + "?"])[sorted_scores]
                                                                 )})
    

    resultsbarcont = st.container()
    


    
    slidercols = st.columns([8,2])
    [highr,lowr] = slidercols[0].select_slider("Rank: select x-axis range",options=[n] + [x*20 for x in range(1,n//20 + 1)][::-1] + [1],value=[100,1])
    timenow = datetime.datetime.utcnow()  




    slidercols[1].download_button(
        label = "Download results",
        data=convert_df(df_occupdata_a),
        file_name=attribute + 'pooled_training_set_results_' + str(timenow.year) + '_' + str(timenow.month) + '_' + str(timenow.day) + '.csv',
        mime='text/csv',
        )    
    st.write("---")    
    setthreshcols = st.columns([3,8])
    threshscore = setthreshcols[0].number_input("Threshold score",min_value=0.,max_value=1.,value=0.5,step=.01)
    threshscoreind = int(threshscore*100)
    setthreshcols[1].write(f"""| Sensitivity (aka Recall) | Specificity | Precision |
| --- | --- | --- |
| {sens[threshscoreind]:.1%} | {spec[threshscoreind]:.1%} | {prec[threshscoreind]:.1%} |
|  {truepos[threshscoreind]} true positives identified out of {realpos} total positives  | {trueneg[threshscoreind]} true negatives identified out of {realneg} total negatives | {truepos[threshscoreind]} true positives out of {idpos[threshscoreind]} positives identified |  
""",unsafe_allow_html=True)

    showresults_chart = alt.Chart(df_sorted_occupations.loc[lowr-1:highr-1],title="Estimated " + attribute + " Scores, in Rank Order")

    @st.cache(ttl=3600)
    def resultsbarchart(basechart,hlelement,theattr):
        return showresults_chart.mark_bar(opacity=.8,width=max(int(10*50/(highr-lowr)),0.5))\
                                           .encode(x=alt.X(theattr + "_rank",
                                                           axis=alt.Axis(grid=False),
                                                           scale=alt.Scale(reverse=True,
                                                           domain=[lowr-0.5,highr+0.5],
                                                           clamp=True,nice=False)),
                                                   y=alt.Y(theattr + "_score",
                                                           axis=alt.Axis(grid=False)),
                                                   tooltip=[alt.Tooltip("occup_title",title="Occupation"),
                                                            alt.Tooltip(theattr + "_rank",title="Rank"),
                                                            alt.Tooltip(theattr + "_score",title="Score",format='.2f')],
                                                   color=alt.Color(theattr + "?",scale=alt.Scale(domain=['YES','NO','Not classified'],
                                                                                                   range=['green','red','lightgrey'])),
                                                   opacity=alt.condition(~hlelement,alt.value(.9),alt.value(.5))).properties(
                                                        width=600,height=300
                                            ).add_selection(hlelement)
    showresults_bar = resultsbarchart(showresults_chart,highlighter,attribute)
    cutoff = alt.Chart(pd.DataFrame({'y':[threshscore]})).mark_rule(color='blue',strokeDash=[5,3],opacity=.7).encode(y='y')
    resultsbarcont.altair_chart(showresults_bar + cutoff)
    




with finaldiv:
    attrselectcols2 = st.columns(2)
    attrlist = list(pd.unique(df_classdata["attribute"].sort_values()))
    firstattr = attrselectcols2[0].selectbox("First attribute to analyze:",attrlist)
    attrlist2 = []
    for attr in attrlist:
        if attr != firstattr:
            attrlist2.append(attr)

    secondattr = attrselectcols2[1].selectbox("Second attribute to analyze:",attrlist2)


    st.write("###### Which principal components to include:")
    PC_cb_cols2 = st.columns(10)
    st.write(" ")

    ordinalnums = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th"]
    defaults = [True,True,True,True,True,False,False,False,False]
    includePC_2 = [False]*9
    for j in range(9):
        includePC_2[j] = PC_cb_cols2[j+1].checkbox(ordinalnums[j],value=defaults[j],key=20+j)



    scattercols = st.columns([8,3])
    firstthresh = scattercols[1].number_input("Threshold score, " + firstattr,min_value=0.,max_value=1.,value=0.5,step=.01)
    secondthresh = scattercols[1].number_input("Threshold score, " + secondattr,min_value=0.,max_value=1.,value=0.5,step=.01)


    #df_pcomp_2 = pd.DataFrame(data=TT,columns=["PC" + str(j+1) for j in range(k)])
    #df_pcomp_2["O*NET-SOC Code"] = np.array(df_occupdata["O*NET-SOC Code"])
    
    df_curdata_1 = df_curdata = build_dataset(TT,k,df_occupdata,df_classdata,firstattr)
    
    #df_curdata_1 = df_classdata.loc[np.array(df_classdata["attribute"]) == firstattr].merge(df_pcomp_2,how="left",left_on="ONET_string",right_on="O*NET-SOC Code")
    _,_,yhat1,_,_,_,_,_,_,_,_,_,_,_ = cached_logit(df_curdata_1,includePC_2,TT)

    sorted_scores1 = np.argsort(yhat1)[::-1]
    df_results1 = pd.DataFrame({firstattr + "_score":yhat1.flatten()[sorted_scores1],
                                              firstattr + "_rank":[x + 1 for x in list(range(n))],
                                              "occup_title":list(
                                                                 np.array(
                                                                        df_occupdata['Title']
                                                                        )[sorted_scores1]
                                                                     )})
    
    df_curdata_2 = df_curdata = build_dataset(TT,k,df_occupdata,df_classdata,secondattr)
    
    #df_curdata_2 = df_classdata.loc[np.array(df_classdata["attribute"]) == secondattr].merge(df_pcomp_2,how="left",left_on="ONET_string",right_on="O*NET-SOC Code")
    _,_,yhat2,_,_,_,_,_,_,_,_,_,_,_ = cached_logit(df_curdata_2,includePC_2,TT)

    sorted_scores2= np.argsort(yhat2)[::-1]
    df_results2 = pd.DataFrame({secondattr + "_score":yhat2.flatten()[sorted_scores2],
                                              secondattr + "_rank":[x + 1 for x in list(range(n))],
                                              "occup_title":list(
                                                                 np.array(
                                                                        df_occupdata['Title']
                                                                        )[sorted_scores2]
                                                                     )})

    @st.cache(ttl=3600)
    def make_combined_scatter(results1,results2,thresh1,thresh2,attr1,attr2,hlelement):
        combined_results = pd.merge(results1,results2,how="inner",on="occup_title")
        combined_results["Classification"] = "None"
        combined_results.loc[(combined_results[attr1 + "_score"] >= thresh1) 
                              & (combined_results[attr2 + "_score"] < thresh2),"Classification"] = attr1 + " only"
        combined_results.loc[(combined_results[attr1 + "_score"] < thresh1) 
                              & (combined_results[attr2 + "_score"] >= thresh2),"Classification"] = attr2 + " only"
        combined_results.loc[(combined_results[attr1 + "_score"] >= thresh1) 
                              & (combined_results[attr2 + "_score"] >= thresh2),"Classification"] = \
                                    "Both"


        return alt.Chart(combined_results,width=400,height=350).mark_point().encode(x=alt.X(attr1 + "_rank",scale=alt.Scale(reverse=True)),
                                                                                          y=alt.Y(attr2 + "_rank",scale=alt.Scale(reverse=True)),
                                                                      tooltip=[alt.Tooltip("occup_title",title="Occupation"),
                                                                               alt.Tooltip(attr1 + "_score",title=attr1 + " score",format='.3f'),
                                                                               alt.Tooltip(attr2 + "_score",title=attr2 + " score",format='.3f')],
                                                                      opacity=alt.condition(~hlelement,alt.value(0.5),alt.value(0.9)),
                                                                      color=alt.Color("Classification",
                                                                                      scale=alt.Scale(domain=["None",
                                                                                                              attr1 + " only",
                                                                                                              attr2 + " only",
                                                                                                              "Both"])))\
                                                                     .add_selection(hlelement).interactive()
    
    scattering = make_combined_scatter(df_results1,df_results2,firstthresh,secondthresh,firstattr,secondattr,highlighter)

    #combined_results = pd.merge(df_results1,df_results2,how="inner",on="occup_title")
    #combined_results["Classification"] = "None"
    #combined_results.loc[(combined_results[firstattr + "_score"] >= firstthresh) 
    #                      & (combined_results[secondattr + "_score"] < secondthresh),"Classification"] = firstattr + " only"
    #combined_results.loc[(combined_results[firstattr + "_score"] < firstthresh) 
    #                      & (combined_results[secondattr + "_score"] >= secondthresh),"Classification"] = secondattr + " only"
    #combined_results.loc[(combined_results[firstattr + "_score"] >= firstthresh) 
    #                      & (combined_results[secondattr + "_score"] >= secondthresh),"Classification"] = \
    #                            "Both"
    #
    #
    #scattering = alt.Chart(combined_results,width=480,height=420).mark_point().encode(x=alt.X(firstattr + "_rank",scale=alt.Scale(reverse=True)),
    #                                                                                  y=alt.Y(secondattr + "_rank",scale=alt.Scale(reverse=True)),
    #                                                              tooltip=[alt.Tooltip("occup_title",title="Occupation"),
    #                                                                       alt.Tooltip(firstattr + "_score",title=firstattr + " score",format='.3f'),
    #                                                                       alt.Tooltip(secondattr + "_score",title=secondattr + " score",format='.3f')],
    #                                                              opacity=alt.condition(~highlighter,alt.value(0.5),alt.value(0.9)),
    #                                                              color=alt.Color("Classification",
    #                                                                              scale=alt.Scale(domain=["None",
    #                                                                                                      firstattr + " only",
    #                                                                                                      secondattr + " only",
    #                                                                                                      "Both"])))\
    #                                                             .add_selection(highlighter).interactive()


    scattercols[0].altair_chart(scattering)



#####


methodology.write("""
This app was developed by [Matt Delventhal](https://mattdelventhal.com/ "Matt Delventhal's website"). It uses information provided by the 
U.S. Department of Labor's O*NET program, combined with a database of human classifications, to assign attributes to occupations.

Users of the companion [Occupation classifier](https://mattdelventhal.com/project/occu_classify/ "Occupation classifier") app have classified subsets of occupations as either having, or not, certain 
attributes--for example, is Chief Executive Officer a "Prestigious" occupation, or not? Every hour, the app retrieves the sum of these human 
classifications from a live database to use as a training set. It then fits a logistic regression to this set using principal components of an array 
of 353 O*NET occupation characteristics. This produces a score between 0 and 1. Strictly speaking, this represents the estimated probability that a 
given occupation will possess the given attribute. We can also think of this score as representing how *intensely* the occupation 
possesses that attribute (e.g. *how* Analytical is it?).

In what follows we will briefly discuss some details of the logistic regression, some particular nuances of the calculations 
reflected in the **Classify one attribute** tab, and explain how O*NET occupation characteristic data is selected and summarized in principal 
components.

##### Logistic regression

A simple unpenalized logistic regression is used. In addition to the principal components which are selected by you, the user, a full set of 
submission-level fixed effects are included. This allows for the possibility that different human contributors, at different moments, 
may have differences in personality and mood which lead to systematic differences in how likely they are to click "Yes" or "No" for 
a particular attribute.

To make projections onto the full set of occupations, the coefficients of these fixed effects are combined into a single constant term 
by taking the weighted average, weighted by number of occupations classified.

##### Classify one attribute tab 

In the bar chart of classification results, bars are colored as follows: If there are more "Yes" classifications for that occupation in 
the training data than "No," it is colored green. If there are more "No"'s than "Yes"'s, it is colored red. If no one has classified that 
occupation, and in a few rare cases where it has been classified but "No"s and "Yes"s are evenly balanced, the corresponding bar is left 
light gray.

The "Sensitivity," "Specificity," "Precision," and "Recall" statistics, and the raw numbers in the explanations below them, are based on the 
total number of classifications in the training set. If an occupation is rated "Yes" twice, that counts as two positives in the data. This 
can lead to small discrepancies between the statistics reported in the table and the values it appears they should take, based on the colored 
bar chart above it.

##### O*NET Data

[O*NET Online](https://www.onetonline.org/ "O*NET Online") provides data on a wide range of occupation characteristics, which they have 
produced through a combination of surveys and expert analysis. This includes information on what kinds of skills are required to succeed 
in the occupation and what kinds of tasks workers in the occupation usually perform. This app uses data from the "Abilities," "Interests," 
"Knowledge," "Skills," "Work Activities," "Work Styles," and "Work Values" tables. These provide data for 873 occupations. Any variable which 
is missing for any of these occupations is excluded from our analysis. Some of the tables, such as the "Skills" table, provide information 
on both how important a skill is, and what level of skill is required. In those cases, both "importance" and "level" for that particular skill 
are registered as distinct variables. All variables are normalized linearly to take values between 0 and 1. The final set includes 353 
occupation characteristics.

You may download the dataset and the variable descriptions by clicking the buttons below."""
)
download_columns = methodology.columns(2)
download_columns[0].download_button(
        label = "Download dataset",
        data=convert_df(df_occupdata),
        file_name='occup_data_toanalyze.csv',
        mime='text/csv',
        )
download_columns[1].download_button(
        label = "Download variable descriptions",
        data=convert_df(df_occupvarinfo),
        file_name='occup_varnames.csv',
        mime='text/csv',
        )


methodology.write("""
The data is then arranged in an 873 by 353 matrix, where each column represents an occupation characteristic. The column-wise mean is 
subtracted from each column, so that each resulting column has a mean of 0. Then the eigenvectors of the resulting matrix are taken as 
the weight vectors defining the dataset's principal components, as is standard. 
[Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis "Principal Component Analysis") provides a useful description 
of this standard statistical technique.
""")

screeplot = alt.Chart(pd.DataFrame({"component_number":[x+1 for x in list(range(len(vv[0:40])))],"variation_explained":vv[0:40]/np.sum(vv)}))
screedots = screeplot.mark_point().encode(y=alt.Y("variation_explained",axis=alt.Axis(format=".0%")),x="component_number",
            tooltip=[alt.Tooltip("component_number"),alt.Tooltip("variation_explained",format=".1%")],
            size=alt.condition(~highlighter,alt.value(20),alt.value(45)))\
            .add_selection(highlighter).interactive()
screelines = screeplot.mark_line().encode(y="variation_explained",x="component_number")


cumscreeplot = alt.Chart(pd.DataFrame({"component_number":[x+1 for x in list(range(len(vv[0:40])))],"cumulative_variation_explained":np.cumsum(vv[0:40]/np.sum(vv))}))
cumscreedots = cumscreeplot.mark_point().encode(y=alt.Y("cumulative_variation_explained",axis=alt.Axis(format=".0%")),x="component_number",
            tooltip=[alt.Tooltip("component_number"),alt.Tooltip("cumulative_variation_explained",format=".1%")],
            size=alt.condition(~highlighter,alt.value(20),alt.value(45)))\
            .add_selection(highlighter).interactive()
cumscreelines = cumscreeplot.mark_line().encode(y="cumulative_variation_explained",x="component_number")


screeoption = methodology.radio("Show:",['Variance explained','Cumulative variance explained'],index=0)
if screeoption == 'Variance explained':
    methodology.altair_chart((screedots + screelines))
elif screeoption == 'Cumulative variance explained':
    methodology.altair_chart((cumscreedots + cumscreelines))

methodology.write("""
The chart above shows the fraction of variance explained by each principal component. We use only the first two because experimentation 
showed that more than this can lead to severe over-fitting in small samples, while one component alone is often not enough to achieve 
a good classification.

We can get some idea of what each principal component represents by looking at the biggest weights for each. Click the boxes below
to show or hide the 
10 highest-weighted occupation characteristics for each of the first nine components. As can be inferred from the previous chart, these 
components together account for almost two thirds of all the variation in the data.
""")

df_varinfo_copy = df_occupvarinfo.copy()
if methodology.checkbox("Top ten weights for first component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,0])[::-1]
    df_varinfo_copy["PC_1_weights"] = WW[:,0]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_1_weights"]])

if methodology.checkbox("Top ten weights for second component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,1])[::-1]
    df_varinfo_copy["PC_2_weights"] = WW[:,1]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_2_weights"]])

if methodology.checkbox("Top ten weights for third component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,2])[::-1]
    df_varinfo_copy["PC_3_weights"] = WW[:,2]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_3_weights"]])

if methodology.checkbox("Top ten weights for fourth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,3])[::-1]
    df_varinfo_copy["PC_4_weights"] = WW[:,3]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_4_weights"]])

if methodology.checkbox("Top ten weights for fifth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,4])[::-1]
    df_varinfo_copy["PC_5_weights"] = WW[:,4]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_5_weights"]])

if methodology.checkbox("Top ten weights for sixth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,5])[::-1]
    df_varinfo_copy["PC_6_weights"] = WW[:,5]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_6_weights"]])

if methodology.checkbox("Top ten weights for seventh component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,6])[::-1]
    df_varinfo_copy["PC_7_weights"] = WW[:,6]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_7_weights"]])

if methodology.checkbox("Top ten weights for eighth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,7])[::-1]
    df_varinfo_copy["PC_8_weights"] = WW[:,7]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_8_weights"]])

if methodology.checkbox("Top ten weights for ninth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,8])[::-1]
    df_varinfo_copy["PC_9_weights"] = WW[:,8]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_9_weights"]])



    

