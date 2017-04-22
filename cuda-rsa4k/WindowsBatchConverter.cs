using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeviceBuildScript
{
    class WindowsBatchConverter
    {
        // statics
        private static string COMMENT = "#$";
        private static string SET = "set ";
        private static string CALL = "call \"";
        private static string VSVARS = "vcvars";
        private static string DEVICE_WRAPPER = "DeviceWrapper.cu";
        private static string ECHO_OFF = "@echo off";
        private static string CICC = "cicc";
        private static string OUTPUT = "DeviceWrapper.ptx";        
        private static string DUMMY_OUTPUT = "DummyDeviceWrapper.ptx";  

        static void Main(string[] args)
        {
            //buffer
            string[] text = new string[1024];

            if (args.Length < 1)
                return;

            // Read the file and display it line by line.
            System.IO.StreamReader file = new System.IO.StreamReader(args[0]);
            string line;
            int counter = 0;
            int commandsOffset = -1;
            while ((line = file.ReadLine()) != null)
            {
                if (line == DEVICE_WRAPPER)
                    commandsOffset = counter;
                string firstWord = line.Split(' ')[0];
                if (firstWord == COMMENT)
                    text[counter++] = line.Remove(0, 3);                
            }      
            file.Close();
            
            for (int i = 0; i < commandsOffset; i++)
            {
                if (text[i].Contains(VSVARS))
                    text[i] = CALL + text[i] + "\"";
                else
                    text[i] = SET + text[i];
            }

            for (int i = 0; i < counter; i++)
            {
                if (text[i].Contains(CICC)) 
                {
                    text[i] = text[i].Replace(OUTPUT, DUMMY_OUTPUT);
                    break;
                }
            }

            Console.WriteLine(ECHO_OFF);
            for (int i = 0; i < counter; i++)
            {
                Console.WriteLine(text[i]);
            }            
        }
    }
}
