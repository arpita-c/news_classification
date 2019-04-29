println("Spark version = " + sc.version)
val sqlContext= new org.apache.spark.sql.SQLContext(sc)
println("Spark SQL context: " + sqlContext)
import sqlContext.implicits._

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, IDF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector

val path = "mini_newsgroups/*"
val newsgroupsRawData = sc.wholeTextFiles(path)
println("The number of documents read in is " + newsgroupsRawData.count() + ".")
newsgroupsRawData.takeSample(false, 1, 10L).foreach(println)
val filepath = newsgroupsRawData.map{case(filepath,text) => (filepath)}

filepath.takeSample(false, 5, 10L).foreach(println)
val text = newsgroupsRawData.map{case(filepath,text) => text}

text.takeSample(false, 1, 10L).foreach(println)
val id = filepath.map(filepath => (filepath.split("/").takeRight(1))(0))
id.take(5).foreach(println)

val topic = filepath.map (filepath => (filepath.split("/").takeRight(2))(0))
topic.distinct().take(20).foreach(println)

case class newsgroupsCaseClass(id: String, text: String, topic: String)

val newsgroups = newsgroupsRawData.map{case (filepath, text) => 
    val id = filepath.split("/").takeRight(1)(0)
    val topic = filepath.split("/").takeRight(2)(0)
    newsgroupsCaseClass(id, text, topic)}.toDF()
newsgroups.cache()

newsgroups.printSchema()
newsgroups.sample(false,0.005,10L).show(5)

newsgroups.groupBy("topic").count().show()
newsgroups.filter(newsgroups("topic").like("comp%")).sample(false,0.01,10L).show(5)

val labelednewsgroups = newsgroups.withColumn("label", newsgroups("topic").like("comp%").cast("double"))
labelednewsgroups.sample(false,0.003,10L).show(5)
labelednewsgroups.filter(newsgroups("topic").like("comp%")).sample(false,0.007,10L).show(5)

val Array(training, test) = labelednewsgroups.randomSplit(Array(0.9, 0.1), seed = 12345)
println("Total Document Count = " + labelednewsgroups.count())
println("Training Count = " + training.count() + ", " + training.count*100/(labelednewsgroups.count()).toDouble + "%")
println("Test Count = " + test.count() + ", " + test.count*100/(labelednewsgroups.count().toDouble) + "%")


val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lr))

println("Logistic Regression Features Column = " + lr.getFeaturesCol)
println("Logistic Regression Label Column = " + lr.getLabelCol)
println("Logistic Regression Threshold = " + lr.getThreshold)


remover.getStopWords.foreach(println)

val model = pipeline.fit(training)
val predictions = model.transform(test)

predictions.select("id", "topic", "probability", "prediction", "label").sample(false,0.01,10L).show(5)
predictions.select("id", "topic", "probability", "prediction", "label").filter(predictions("topic").like("comp%")).sample(false,0.1,10L).show(5)

predictions.sample(false,0.001,10L).show(5)

val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
println("Area under the ROC curve = " + evaluator.evaluate(predictions))

val paramGrid = new ParamGridBuilder().
  //addGrid(hashingTF.numFeatures, Array(1000, 10000, 100000)).
  //addGrid(idf.minDocFreq, Array(0,10, 100)).
  addGrid(lr.regParam, Array(0.01, 0.1, 0.2)).
  addGrid(lr.threshold, Array(0.5, 0.6, 0.7)).
  build()

 val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)

val cvModel = cv.fit(training)
println("Area under the ROC curve for best fitted model = " + evaluator.evaluate(cvModel.transform(test)))

println("Area under the ROC curve for non-tuned model = " + evaluator.evaluate(predictions))
println("Area under the ROC curve for fitted model = " + evaluator.evaluate(cvModel.transform(test)))
println("Improvement = " + "%.2f".format((evaluator.evaluate(cvModel.transform(test)) - evaluator.evaluate(predictions)) *100 / evaluator.evaluate(predictions)) + "%")

cvModel.bestModel.stages(4)
cvModel.avgMetrics


cvModel.transform(test).select("id", "topic", "probability", "prediction", "label").sample(false,0.01,0L).show(5)
cvModel.transform(test).select("id", "topic", "probability", "prediction", "label").filter(predictions("topic").like("comp%")).sample(false,0.1,0L).show(5)
