# Predict Disease Based On Symptoms


Abstract:
With the rise in the number of patients and diseases, every year the medical system is overloaded and with time it has become overpriced in many countries. Most of the disease involves consultation with doctors to get treated. Many people are looking online for health information regarding diseases, diagnoses and different treatments. If a recommendation system can be made for doctors and Patients will save a lot of time. With sufficient data, prediction of disease by an algorithm can be very easy and cheap.
In this project, the model is trained to predict a disease based on symptoms of the patient. Here an interactive interface is designed to facilitate interaction with the system. An interactive system displays a web page where the patient must fill in his details and enter the symptoms. The model is trained using two different algorithms: Random Forest, Decision tree predicts the disease and displays it on the web page.
With the growth of medical data, many researchers are using these medical data and some machine learning algorithms to help the healthcare communities in the diagnosis of many diseases. In this paper a survey of various models based on such algorithms and techniques is presented and their performance is analyzed.




Introduction:

Human life and the economy are greatly impacted by medicine and the health industry. It is a major concern everywhere that public health is given more importance since every nation wants to prevent diseases that could increase the state's mortality rate. Everything has changed to be horrifying and bizarre. The doctors and nurses are working as hard as they can to save people's lives in this situation where everything has gone virtual, even at the risk of putting their own lives in risk. Additionally, some isolated areas lack access to healthcare. Virtual doctors are board-certified physicians who prefer to conduct business over the phone and via video consultations rather than in-person visits; however, this is not an option in an emergency. Machines are always viewed as having a faster and more reliable technique of completing tasks than individuals because there is no space for human error. One could refer to a disease predictor as a virtual physician.


The ability of machine learning techniques to totally change how diseases are anticipated and diagnosed has recently attracted a lot of interest in the medical community. The purpose of this literature review is to provide a thorough overview of machine learning-based disease prediction. Analysis of the methodology, datasets, and results used in this research is the main objective in order to gain knowledge of the state of the subject now. This review will give a comprehensive knowledge of the successes, constraints, and opportunities for development in the use of machine learning to illness prediction by synthesizing and summarizing the important findings from the literature. Machine learning techniques are used for prediction in a variety of fields since the findings are consistent and dependable. In this work, multiple machine learning techniques are examined and presented for identifying high risk diseases.

Research Objective:
The main objective of this paper is to identify the various findings and gaps which can be further explored by researchers. The following are objectives of this literature survey. 
(1) To develop a strong background on health care domain.
(2) To come up with a new problem which is not addressed. 
(3) To identify the research gap. 
(4) To design a new solution for solving the identified problem.



Purpose:
The purpose of this literature review is to critically examine and analyze the research in the area of Disease Prediction Using Machine Learning. Identification of most common themes, trends and methodologies used in the field. This review lays a strong foundation that helps researchers, healthcare professionals, and policymakers with a deep insight of the contemporary landscape of disease prediction. This landscape makes use of machine learning (ML), utilizing a wide array of healthcare data sources and advanced algorithms to improve disease prediction. The primary purpose of this literature review is to provide a comprehensive understanding of disease prediction using machine learning. The machine learning algorithms that are often employed in illness prediction will be covered in this part. The model is trained using two different algorithms: Random Forest, Decision tree predicts the disease and displays it on the web page.

Methodology:
Importance of Machine learning:
The machine learning field is continuously evolving. And along with evolution comes a rise in demand and importance. There is one crucial reason why data scientists need machine learning, and that is: ‘High-value predictions that can guide better decisions and smart actions in real-time without human intervention.’
Machine learning as technology helps analyze large chunks of data, easing the tasks of data scientists in an automated process and is gaining a lot of prominence and recognition. Machine learning has changed the way data extraction and interpretation works by involving automatic sets of generic methods that have replaced traditional statistical techniques.

ALGORITHM MODELS:
There are different kinds of models used to predict the disease and they are:
•	Decision tree
•	Random forest tree
•	      KNN
Decision Tree:
Decision tree is classified as a very effective and versatile classification technique. It is used in pattern recognition and classification for images. It is used for classification in very complex problems due to its high adaptability. It is also capable of engaging problems of higher dimensionality. It mainly consists of three parts root, nodes and leaf.
Roots consists of attribute which has most effect on the outcome, leaf tests for value of certain attribute and leaf gives out the output of tree.
Decision Tree is a supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules, and each leaf node represents the outcome.
In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
The decisions or the test are performed based on features of the given data set.
The Logic behind the decision tree can be easily understood because it shows a tree-like Structure.

Random-Forest Algorithm:
Random Forest Algorithm is a supervised learning algorithm used for both classification and regression.
This algorithm works on 4 basic steps –
1.       It chooses random data samples from dataset.
2.       It constructs decision trees for every sample dataset chosen.
3.       At this step every predicted result will be compiled and voted on.
4.       Finally, most voted predictions will be selected and be presented as result of classification.

One big advantage of random forests is that they can be used for both classification and Regression problems which form many current machine learning systems.
Random forest algorithm creates decision trees on data samples and gets the prediction from each of them and finally selects the best solution by means of voting. It is an
The Ensemble method is better than a single decision tree because it reduces the over -Fitting by averaging the result.
It takes less training time as compared to other algorithms. It predicts output with high accuracy, even for the large dataset it runs efficiently. It can also maintain accuracy when a large proportion of data is missing.

KNN:

K-Nearest Neighbors (KNN) is a straightforward, instance-based, and non-parametric classification technique. Regression activities can also be performed using it. The core concept of KNN is to forecast the class of a new data point using the majority class of its K nearest neighbors. Here are how the algorithm functions:

1. The Training Phase:
The method merely memorizes the full training dataset throughout the training phase. Instead of creating an explicit model, it keeps all the training examples.

2. Prediction Phase: The algorithm determines the distance between a new data point and each other point in the training dataset before predicting the class of that point. Although Euclidean distance is frequently employed, other distance metrics, such as Manhattan, Minkowski, etc., can also be utilized depending on the type of data.
The top K nearest data points—those with the shortest distances from the new point—are then chosen by the algorithm.
KNN uses a majority vote among the K neighbors for classification jobs. The new data point is given the class that appears the most frequently among its K nearest neighbors. When performing regression, the technique averages (or weights) the values of the K closest neighbors.

3. Choosing the Value of K: K must be chosen carefully. A lower K value results in noisy output, whereas a higher K value increases the computing cost of the process. Cross-validation methods, for example, can be used to find the ideal K.
4. Distance Metrics: A distance measure should be chosen based on the work at hand and the type of data. Although the curse of dimensionality may prevent it from working with high-dimensional data, Euclidean distance is the most widely used alternative.  Other distance metrics or dimensionality reduction strategies are applied in these circumstances.
In a study, Decision tree classifier is applied. This classifier appears to be using recursive splitting of the sample space [5]. The predictive approach here functions as a mapping between the object's characteristics and values [6]. A two-phase support vector network was used which combined the two-phase clustering approach with a probability based SVM that evaluated the Wisconsin Breast Cancer Diagnosis (WBCD) dataset [7]. It achieved a classification model accuracy of 99.10 percent.
Karayilan et al. [8] proposed a heart disease prediction system that uses the artificial neural network backpropagation algorithm. 13 clinical features were used as input for the neural network and then the neural network was trained with the backpropagation algorithm to predict absence or presence of heart disease with an accuracy of 95 %. Various machine learning algorithms were streamlined for the effective prediction of a chronic disease outbreak by Chen et al. [9]. The data collected for the training purpose was incomplete. To overcome this, a latent factor model was used. A new convolutional neural network-based multimodal disease risk prediction (CNN-MDRP) was structured. The algorithm reached an accuracy of around 94.8 %.

Approach:
Machine learning algorithms like Random Forest and Decision Trees require a systematic method to handle complicated and variable medical data in order to predict diseases based on symptoms. First, it is necessary to curate extensive and trustworthy databases that include symptoms and the diseases they correspond to. To manage missing values, encode category variables, and standardize the data for consistency, these datasets need to be preprocessed. In order to ensure that the model concentrates on the critical elements for precise predictions, feature selection approaches are used to determine the most pertinent symptoms. The performance of the models may then be assessed thanks to the division of the data into training and testing sets.

Proposed method for predicting illness. There's a chance the doctor won't always be on call. However, one can always apply this prediction mechanism whenever necessary in the present world. An individual's symptoms, together with their age and gender, can be provided to the ML model for additional processing. The machine learning model uses the most recent input, trains, and tests the algorithm to forecast the disease after basic data processing.

Decision trees are built to link symptoms to diseases, giving the decision-making process clear knowledge. Random Forests are used, though, to improve accuracy and robustness. By training on different subsets of the data, Random Forests produces an ensemble of Decision Trees. These ensembles then make predictions by combining the outputs of several trees. In circumstances where individual Decision Trees might overfit the training data, this ensemble technique increases the model's prediction ability. For the Random Forest model to function at its best and be a trustworthy tool for disease prediction based on symptoms, proper parameter tuning and evaluation approaches, such as cross-validation, are essential.


To summarize, the procedure entails gathering data, preprocessing, choosing features, separating the data, building Decision Trees to comprehend the links between symptoms and diseases, and then utilizing Random Forests to create a reliable predictive model. This strategy enables machine learning algorithms to assess symptoms and forecast diseases, assisting medical practitioners in making accurate diagnoses and implementing appropriate therapies.

Research Gaps:
Here are three research holes and succinct recommendations for symptom-based prediction using machine learning:
Research Gap 1: Limited generalizability 
The following is a proposal to improve generalizability: gather various and representative datasets, investigate transfer learning, and evaluate models in various clinical situations.
Research Gap 2: Explainability and Interpretability
To improve predictability, it is suggested that interpretable machine learning models be created, feature importance analysis be used, and model-agnostic interpretability techniques be used.
Research Gap 3: Ethics and Data Privacy
To address privacy and ethical issues with symptom-based prediction, a proposal has been made to investigate privacy-preserving machine learning approaches, establish explicit data sharing and consent mechanisms, and assure ethical treatment of sensitive healthcare data.

Challenges and Limitations: 
In this part, we will examine the difficulties and constraints that machine learning researchers in illness prediction must overcome, such as data privacy concerns, model interpretability, and bias.

Below are challenges faced for developing Disease prediction using Machine Learning		
•	Data imbalance: Medical datasets frequently experience class imbalance, where some diseases may have a disproportionately low prevalence compared to others, leading to inaccurate predictions.
•	Relevant Features: It is essential to choose the appropriate collection of features (variables) from a range of relevant factors. Selecting redundant or irrelevant features can have a negative effect on the model's performance.
•	Temporal Dynamics: Conditions change with time. Predictive models must take into account the temporal elements of the data, taking into account the progression of diseases and the evolution of patient data.




 Future Scope:
1. Improvements in comprehensible AI for model transparency and trust.
2. Using IoT and wearables, real-time monitoring is being expanded.
3. Responsible AI deployment procedures and regulatory compliance.

Conclusion:
This model predicts the disease based on symptoms given to it. Such a system can decrease the rush at OPDs of hospitals and reduce the workload on medical staff. After studying these methods, it has been found that if we have a structured dataset then the accuracy of prediction is improved. If we can collect millions of structured datasets for a particular disease, then that disease can be predicted with the highest accuracy and data mining can help us collect such datasets. 
The performance of these prediction systems can be enhanced, and the system's scalability and accuracy can be increased, by utilizing the many opportunities that lie ahead. The following study possibilities can be carried out in the future since it is not possible to examine all the options in the time available. Multiple classification techniques and regression techniques should be combined, different types of decision trees and neural networks should be used to check how much accuracy has been improved.













 Reference:
[1]   S. Leoni Sharmila, C. Dharuman and P. Venkatesan, "Disease Classification Using Machine 
        Learning Algorithms - A Comparative Study", International Journal of Pure and Applied 
        Mathematics, vol. 114, no. 6, pp. 1-10, 2017.
[2]   Pingale, K., Surwase, S., Kulkarni, V., Sarage, S., & Karve, A. (2019). Disease Prediction 
        using Machine Learning.
[3] Decision tree and Random Forest algorithm
       https://towardsdatascience.com/decision-tree-and-random-forest-explained- 8d20ddabc9dd
[4] Dataset for this project was collected from a study of university of Columbia performed at   
      New York Presbyterian Hospital during 2004. Link of dataset is given below.                
      http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index. html
[5] Mohammed, M. A., Abd Ghani, M. K., Hamed, R. I., & Ibrahim, D. A. (2017). Analysis of 
      electronic methods for nasopharyngeal carcinoma:  Prevalence, diagnosis, challenges and   
      technologies. Journal of Computational Science, 21(1), 241-254. 
 [6] DeMántaras, R.  L.  (1991).  A distance-based attribute selection measure for decision tree 
      induction. Machine learning, 6(1), 81-92. 
[7] Tedeschi, P., & Sciancalepore, S. (2019, June). Edge and fog computing in critical 
      infrastructures: analysis, security threats, and research challenges.  In 2019 IEEE European 
      Symposium on Security and Privacy Workshops (EuroS&PW) IEEE, 3(2), 1-10.
[8] M. Chen, Y. Hao, K. Hwang, L. Wang, L. Wang, Disease prediction by machine learning over 
      big data from healthcare communities, Ieee Access 5, 8869 (2017).
[9] S. Chae, S. Kwon, D. Lee, predicting infectious disease using deep learning and big data,                               
      international journal of environmental research and public health 15(8), 1596 (2018).




