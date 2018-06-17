library(glmnet)

## ##############################################################
# function computing ...
###############################################################

computeLasso = function(X, Y){


  fit = cv.glmnet(as.matrix(X), Y, nfolds=100)
  found = which(fit$glmnet.fit$beta[,which(fit$lambda == fit$lambda.min)] !=0)

  #remove(fit)
  gc()

  return(found)


}


###############################################################
# function computing ...
###############################################################

generateBoostrapFiles= function(data, Y, noOfBoostraps ){
  
  for (i in 1:noOfBoostraps){
    s=sample(c(1:dim(data)[1]), dim(data)[1], replace = FALSE, prob = NULL)
    saveRDS(data[s,], paste("x", i, ".RData", sep=""))
    saveRDS(Y[s], paste("Y", i, ".RData", sep=""))
  }
  
}


###############################################################

###############################################################


worker1 =  function(id){
  if(id%%9==0){print(paste("boostrap no ",id))}
  bdata=readRDS(paste("/x", id, ".RData", sep=""))
  bY = readRDS(paste("/Y", id, ".RData", sep=""))
  found = computeLasso(bdata,bY)
  return(found)
}

bolasso =  function(data, noOfSamplesV, Y, noOfBootstraps){
  
  results = list()
  print(noOfSamplesV)
  exportCluster = c("worker1", "computeLasso", "cv.glmnet")
  numWorkers <- 2
  
  for (j in 1:length(noOfSamplesV)){
    #print(noOfSamplesV[j])
    bdata = data[1:noOfSamplesV[j],]
    print(c("Dim bdata =", dim(bdata)))
    bY = Y[1:noOfSamplesV[j]]
    result = rep(0, dim(bdata)[2])
    print("making cluster")
    cl <- makeCluster(numWorkers, type = "PSOCK")
    print("exporting cluster")
    clusterExport(cl,exportCluster)
    print("calling cluster")
    res <- parLapply(cl, c(1:noOfBootstraps), worker1)
    stopCluster(cl)
    
    for( i in 1:length(res)){ # counting number of votes for each variable
      result[res[[i]]] = result[res[[i]]] +1
    }
    
    object = list(noOfSamples=noOfSamplesV[j], r=result)
    results[[j]] = object
    gc()
    
  }
  return(results)
}



###############################################################
# function computing ...
###############################################################

calculatePredictionErrorCV = function(data, Y){
  
  k=300
  error = 0
  u=0
  noSamples = dim(data)[1]
  if(is.null(noSamples)){
    noSamples = length(data)
    u=1
  }

  for(i in 1:(k)){

    s1=sample(c(1:noSamples), 0.7*noSamples, replace = FALSE, prob = NULL)
    s2=setdiff(c(1:noSamples),s1)

    if(u==0){
      trainData = data[s1,]
      trainData_labels = Y[s1]
      validData  = data[s2,]
      validData_labels = Y[s2]
    }
    else{
      trainData = data[s1]
      trainData_labels = Y[s1]
      validData  = data[s2]
      validData_labels = Y[s2]
    }
    

    fit = calculatePvalue(trainData, trainData_labels)
    
    
    fit$coeff[is.na(fit$coeff)]=0
    
    
    pred= as.matrix(cbind(1,validData))%*%fit$coeff
    
    err=sum(( pred[,1] - validData_labels)^2)
    
    error = error + err
    
  }
  return(sqrt(error)/ (k*length(validData_labels)))
}

###############################################################
# function computing ...
###############################################################

calculatePvalue = function(d, Y){
  if(!is.null(dim(d))){
    formula1=paste(colnames(d),"+",collapse="")
    formula1=substring(formula1,1,nchar(formula1)-1)
    formula = paste("Y~",formula1,collapse="")
    lm1=lm(as.formula(formula), data=cbind(d,Y))
  }else{
    lm1=lm(Y~d, data.frame(data=cbind(d,Y)))
  }
  
  return(lm1)
}
###############################################################
# function computing ...
###############################################################
calculateSupportVectorError = function(result, noOfVars){
  
  # sum of squared distance between the support vectors
  error = length(setdiff(c(1:noOfVars), result)) + length(setdiff(result,c(1:noOfVars)))
  return(error)
}

###############################################################
# function computing ...
###############################################################
calculateRelevantSupportVectorError = function(result, noOfVars){
  
  error = length(setdiff(c(1:noOfVars), result))
  return(error)
}
###############################################################
# function computing ...
###############################################################
calculatePredictionError = function(trainData, trainData_labels, validData, validData_labels){
  
  fit = calculatePvalue(trainData, trainData_labels)
  pred= as.matrix(cbind(1,validData))%*%fit$coeff
  err =  sqrt(sum(( pred[,1] - validData_labels)^2))
  std = sqrt(sum(( pred[,1] - validData_labels)^2)/(length(validData_labels)-2))
  return(c(err,std))
}



