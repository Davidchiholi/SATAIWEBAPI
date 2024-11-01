namespace SATAIWEBAPI;

// 1. install the Open AI library
// cmd: dotnet add package Azure.AI.OpenAI --prerelease

using System;
using Azure.AI.OpenAI;
using OpenAI.Chat;
using System.ClientModel;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;


class Program
{
    static void Main(string[] args)
    {
    
        Console.WriteLine("Start AI!");

                
        string apikey = File.ReadAllText(@"aikey.txt");

        string jsonstr = File.ReadAllText(@"Json.txt");


        Item? item = JsonSerializer.Deserialize<Item>(jsonstr);


        //Console.WriteLine(item?.matched_list?[262]);

        string matchStr = "";
        // 1. ask the matching trend 
        if (item?.matched_list is not null) {
            foreach (var matchpc in item.matched_list) {
                matchStr = matchStr + ',' + matchpc;
            }
        }
        string question1 = matchStr + " is the matching percentage on 1/30 interval for user and trainer. Please describe the matching trend. ";
        // 2. ask the body size of user
        string question2 = $"Left Hip and Keen distance is {item?.left_hip_keen_distance}." + 
                            $"Left Keen and Ankle distance is {item?.left_keen_ankle_distance}."+
                            $"Right Hip Keen distance is {item?.right_hip_keen_distance}."+
                            $"Right Keen and Ankle distance is {item?.right_keen_ankle_distance}."+
                            $"Left Shoulder and Elbow distance is {item?.left_shoulder_elbow_distance}."+
                            $"Left Elbow and Wrist distance is {item?.left_elbow_wrist_distance}."+
                            $"Right Shoulder and Elbow distance is {item?.right_shoulder_elbow_distance}."+
                            $"Right Elbow and Wrist distance is {item?.right_elbow_wrist_distance}. Use these information to describe the user body size. ";
        // 3. ask the speed of user
        string question3 = $" The {item?.joint1_name} average speed is {item?.joint1_avg_speed} meter/sec. " + 
                            $" The {item?.joint2_name} average speed is {item?.joint2_avg_speed} meter/sec. " + 
                            $" The {item?.joint3_name} average speed is {item?.joint3_avg_speed} meter/sec. " + 
                            $" The {item?.joint4_name} average speed is {item?.joint4_avg_speed} meter/sec. Describle the user speed."  ;
        // 4. ask the flexibility of user
        string question4 = $" The shoulder turning angle is {item?.shoulder_rom} degree. The elbow turning angle is {item?.elbow_rom} degree." +
                            $" The hip turning angle is {item?.hip_rom} degree. The knee turning angle is {item?.knee_rom} degree. Describle the flexibility of this user. ";
        // 5. ask impulse of user by acceleration
        string question5 = $"The acceleration of {item?.joint1_name} is {item?.joint1_max_acceleration} meter/second. Describle the impulse.";

        Console.WriteLine(question1 + question2 + question3 + question4 + question5);

        //Environment.Exit(0);
        
        AzureOpenAIClient azureClient = new(
            new Uri("https://satmappopenai.openai.azure.com/"),
            new ApiKeyCredential(apikey));
        ChatClient chatClient = azureClient.GetChatClient("satgpt4o");


        ChatCompletion completion = chatClient.CompleteChat(
            [
                // System messages represent instructions or other guidance about how the assistant should behave
                new SystemChatMessage("You are the assistant to describe a user motion on sport. Try to explain the User motion in detial. Do not show original data on reply. Use Traditional Chinese to reply."),
                // User messages represent user input, whether historical or the most recent input
                new UserChatMessage(question1 + question2 + question3 + question4 + question5),
 
            ]);

        Console.WriteLine($"{completion.Role}: {completion.Content[0].Text}");
    }


    public class Item
    {
        public double left_hip_keen_distance { get; set; }
        public double left_keen_ankle_distance { get; set; }
        public double right_hip_keen_distance { get; set; }
        public double right_keen_ankle_distance { get; set; }
        public double left_shoulder_elbow_distance { get; set; }
        public double left_elbow_wrist_distance { get; set; }
        public double right_shoulder_elbow_distance { get; set; }
        public double right_elbow_wrist_distance { get; set; }
       
        public double joint1_avg_speed { get; set; }
        public double joint2_avg_speed { get; set; }
        public double joint3_avg_speed { get; set; }
        public double joint4_avg_speed { get; set; }
        public double shoulder_rom { get; set; }
        public double elbow_rom { get; set; }
        public double hip_rom { get; set; }
        public double knee_rom { get; set; }
        public double joint1_max_acceleration { get; set; }
        public double matched { get; set; }
        public double[]? matched_list {get; set;}

         public string? joint1_name { get; set; }
         public string? joint2_name { get; set; }
         public string? joint3_name { get; set; }
         public string? joint4_name { get; set; }
        
    }
    
}
