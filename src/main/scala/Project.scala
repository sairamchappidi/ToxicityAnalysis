import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.{ IDF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.types.{StructField, StructType, StringType}
import org.apache.spark.sql.functions._


object Project {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "/usr/local/Cellar/hadoop/3.1.1")
    var my_rdd :RDD[String]= null
    val spark: SparkSession = SparkSession.builder()
      .appName("Project")
      .getOrCreate()

    val source = args(0)
//      val source = "./train"
//    val dest = "./out"
    val dest = args(1)

    val sc = spark.sparkContext
    // reading csv dataset
    val schemaString = "id,target,comment_text,severe_toxicity,obscene,identity_attack,insult,threat,asian,atheist,bisexual,black,buddhist,christian,female,   heterosexual,hindu,homosexual_gay_or_lesbian,intellectual_or_learning_disability,jewish,latino,male,muslim,other_disability,other_gender, other_race_or_ethnicity, other_religion, other_sexual_orientation,    physical_disability,  psychiatric_or_mental_illness, transgender, white, created_date, publication_id,parent_id,    article_id rating,funny,wow,sad, likes, disagree, sexual_explicit, identity_annotator_count,  toxicity_annotator_count, wor"
    val schema = StructType(schemaString.split(",").map(fieldName => StructField(fieldName, StringType, true)))

    val rdd = spark.read.format("csv").option("header", "true").schema(schema).csv(source)
    val df = rdd.toDF()
    // dropping null values
    val dfDrop = df.na.drop()

    // mapreduce
    val wordsDF = dfDrop.select(split(dfDrop("comment_text")," ").alias("words"))
    val wordDF = wordsDF.select(explode(wordsDF("words")).alias("word"))
    val wordCountDF = wordDF.groupBy("word").count
    wordCountDF.orderBy(desc("count")).show(20)

    val df2 = dfDrop.selectExpr("cast(target as double) target", "comment_text")

    // converting continuous target label to numerical
    val binarizer: Binarizer = new Binarizer()
      .setInputCol("target")
      .setOutputCol("label")
      .setThreshold(0.5)

    val binarizedDataFrame = binarizer.transform(df2)

    var df3 = binarizedDataFrame.selectExpr("cast(label as Double) label", "comment_text")

    df3 = df3.na.drop()

    // tokenizing the word comments
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")

    // hashing the word comments
    var hashingTF = new org.apache.spark.ml.feature.HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)

    // performing tf-idf
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // splitting the training data
    val Array(train, test) = df3.randomSplit(Array(0.9, 0.1))

    // fitting the logistic regression model
    val lr = new LogisticRegression().setMaxIter(2).setRegParam(0.09).setElasticNetParam(0.9)

    // sending for pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF,idf, lr))

    // grid search
    val paramGrid_lr = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 50))
      .addGrid(lr.regParam, Array(0.01, 0.1))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // cross validation
    val cv_lr = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid_lr)
      .setNumFolds(2)

    val dropTrain = train.na.drop()

    val crossModel_lr = cv_lr.fit(dropTrain)

    val CvtransformModel_lr = crossModel_lr.transform(test)

    val transPrecit = CvtransformModel_lr.selectExpr("cast(label as Double) label", "prediction")

    var lr_accuracy = 0.0
    var lr_preci= 0.0
    var lr_recall = 0.0
    var lr_f1_score = 0.0

    def lr_displayMetrics(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      lr_accuracy = metrics.accuracy
      lr_preci = metrics.weightedPrecision
      lr_recall = metrics.weightedRecall
      lr_f1_score = metrics.weightedFMeasure
    }

    val PredictionAndLabels_lr = transPrecit.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}

    lr_displayMetrics(PredictionAndLabels_lr)

    val svc = new LinearSVC()
      .setMaxIter(2)
      .setRegParam(0.1)

    val pipeline1 = new Pipeline()
      .setStages(Array(tokenizer, hashingTF,idf,svc))

    val paramGrid_svc = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(svc.regParam, Array(0.1, 0.01))
      .build()

    // cross validation
    val cv_svc = new CrossValidator()
      .setEstimator(pipeline1)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid_svc)
      .setNumFolds(2)

    val crossModel_svc =cv_svc.fit(train)

    val CvtransformModel_svc=crossModel_svc.transform(test)

    val transPrecit_svc = CvtransformModel_svc.selectExpr("cast(label as Double) label", "prediction")

    var svc_accuracy = 0.0
    var svc_preci= 0.0
    var svc_recall= 0.0
    var svc_f1_score= 0.0

    def svc_displayMetrics(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      svc_accuracy = metrics.accuracy
      svc_preci = metrics.weightedPrecision
      svc_recall = metrics.weightedRecall
      svc_f1_score = metrics.weightedFMeasure
    }

    val PredictionAndLabels_svc = transPrecit_svc.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    svc_displayMetrics(PredictionAndLabels_svc)

    var output = ""

    output += "Logistic Regression metrics are as below:\n"

    output+= "Logistic Regression accuracy is: \t" + lr_accuracy + "\n"

    output+= "Logistic Regression precision is: \t" + lr_preci + "\n"

    output+= "Logistic Regression Recall is: \t" + lr_recall + "\n"

    output+= "Logistic Regression F1Score is: \t" + lr_f1_score + "\n"

//    output += "Support Vector  metrics are as below:\n"
//
//    output+= "Support Vector accuracy is: \t" + svc_accuracy + "\n"
//
//    output+= "Support Vector precision is: \t" + svc_preci + "\n"
//
//    output+= "Support Vector Recall is: \t" + svc_recall + "\n"
//
//    output+= "Support Vector F1Score is: \t" + svc_f1_score + "\n"
    my_rdd = sc.parallelize(List(output))
    my_rdd.coalesce(1,true).saveAsTextFile(dest)
    sc.stop()

  }
}
