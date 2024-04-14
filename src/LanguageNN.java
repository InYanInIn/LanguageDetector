import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class LanguageNN {
   private double[][] weights;
   private int inputSize;
   private int numLanguages;

   private HashMap<String,ArrayList<String>> textData = new HashMap<>();
   private HashMap<String,ArrayList<Double[]>> inputData = new HashMap<>();

   public LanguageNN(int inputSize, int numLanguages) {
      this.inputSize = inputSize;
      this.numLanguages = numLanguages;
      this.weights = new double[numLanguages][inputSize];
   }

   public int getLanguageIndex(String language) {
      List<String> keys = new ArrayList<>(textData.keySet());
      return keys.indexOf(language);
   }

   public String getLanguage(int index) {
      List<String> keys = new ArrayList<>(textData.keySet());
      return keys.get(index);
   }

   public void train(Double[] input, String language, double learningRate) {
      int languageIndex = getLanguageIndex(language);
      double[] outputs = classifyRaw(input);
      for (int i = 0; i < numLanguages; i++) {
         for (int j = 0; j < inputSize; j++) {
            if (i == languageIndex) {
               weights[i][j] += learningRate * (1 - outputs[i]) * input[j];
            } else {
               weights[i][j] += learningRate * (0 - outputs[i]) * input[j];
            }
         }

         for (int j = 0; i == languageIndex && j < inputSize; j++) {
            weights[i] = normalizeVector(weights[i]);
         }
      }
   }

   public double[] normalizeVector(double[] vector) {
      double magnitude = 0.0;
      for (double component : vector) {
         magnitude += component * component;
      }
      magnitude = Math.sqrt(magnitude);

      double[] normalizedVector = new double[vector.length];
      for (int i = 0; i < vector.length; i++) {
         normalizedVector[i] = vector[i] / magnitude;
      }
      return normalizedVector;
   }

   public int classify(Double[] input) {
      double[] outputs = classifyRaw(input);
      int bestIdx = 0;
      for (int i = 1; i < outputs.length; i++) {
         if (outputs[i] > outputs[bestIdx]) bestIdx = i;
      }
      return bestIdx;
   }

   private double[] classifyRaw(Double[] input) {
      double[] outputs = new double[numLanguages];
      for (int i = 0; i < numLanguages; i++) {
         for (int j = 0; j < inputSize; j++) {
            outputs[i] += weights[i][j] * input[j];
         }
      }
      return outputs;
   }

   public void extractTestFeatures() {
      for (Map.Entry<String, ArrayList<String>> entry : textData.entrySet()) {
         String language = entry.getKey();
         ArrayList<String> texts = entry.getValue();
         for (String text : texts) {
            inputData.putIfAbsent(language, new ArrayList<>());
            inputData.get(language).add(extractFeatures(text));
         }
      }

   }

   public static Double[] extractFeatures(String text) {
      Double[] features = new Double[26];
      Arrays.fill(features, 0.0);

      text = text.toLowerCase();
      int totalLetters = 0;

      for (char c : text.toCharArray()) {
         if (c >= 'a' && c <= 'z') {
            features[c - 'a']++;
            totalLetters++;
         }
      }

      if (totalLetters > 0) {
         for (int i = 0; i < features.length; i++) {
            features[i] /= totalLetters;
         }
      }

      return features;
   }

   public void trainModel(double learningRate) {
      for (Map.Entry<String, ArrayList<Double[]>> entry : inputData.entrySet()) {
         String language = entry.getKey();
         ArrayList<Double[]> features = entry.getValue();
         for (Double[] feature : features) {
            train(feature, language, learningRate);
         }
      }
   }

   public void readLanguageFiles(String directoryPath) {
    Path dirPath = Paths.get(directoryPath);
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath)) {
        for (Path languagePath : stream) {
            if (Files.isDirectory(languagePath)) {
                String language = languagePath.getFileName().toString();
                try (DirectoryStream<Path> languageStream = Files.newDirectoryStream(languagePath)) {
                    for (Path filePath : languageStream) {
                      List<String> lines = Files.readAllLines(filePath);
                      textData.putIfAbsent(language, new ArrayList<>());
                      textData.get(language).add(String.join(" ", lines));

                    }
                }
            }
        }
    } catch (IOException e) {
        System.err.println("Error reading files: " + e);
    }
   }


   public static void main(String[] args) {
      Scanner scanner = new Scanner(System.in);
      LanguageNN nn = new LanguageNN(26, 3); // assuming 3 languages and 26 features

      nn.readLanguageFiles("languages");
      nn.extractTestFeatures();
      nn.trainModel(0.1);




      while (true) {
         System.out.println("Enter text to classify or (q) to exit the programm: ");
         String text = scanner.nextLine();

         if (text.equals("q")) {
            break;
         }

         Double[] input = LanguageNN.extractFeatures(text);
         int languageIndex = nn.classify(input);
         String language = nn.getLanguage(languageIndex);
         System.out.println("The language is: " + language);
      }


   }
}
