#include <Servo.h>

/*
 * command map
 * u --> sistem up (x axis motor up)
 * d --> sistem down (x axis motor down)
 * l --> sistem left (y axis motor left)
 * r --> sistem right (y axis motor right)
 * f --> sistem fire (set lazer up for fire)
 * s --> sistem stop fire (set lazer down to stop fire)
 * c --> sistem initialization initialize sistem (motors on 90 degre and lazer down)
 * to send data to serial monitor you must send it like command_x,command_y,_command_lazer
 */

Servo x_motor; // x axis motor
Servo y_motor; // y axis motor
int lazer = 13; // lazer pin

int initialization_angle = 90; // angle to set motor on 90 degre position
int max_angle = 175; // maximum angle that motor can reach 
int min_angle = 5; //manimum angle that motor can reach 
int angle_x=90; // angle to initialize x_motor to 90 degre
int angle_y=90; // angle to initialize y_motor to 90 degre
int increment_angle = 5; // angle that add or substitute on motor angle to turn it
int fire = 1; // value to set up lazer for fire
int stop_fire = 0; // value to set down lazer to stop fire

String Receive; // value to store receive data like string
char command_x; // command to control x axis motor
char command_y; // command to control y axis motor
char command_l; // command to control lazer

void setup() {
  Serial.begin(9600);
  x_motor.attach(2); //attach x axis motor on pin2 (pwm pin on arduino mega)
  y_motor.attach(3); //attach y axis motor on pin3 (pwm pin on arduino mega)
  pinMode(lazer,OUTPUT); // set lazer pin like outpu

  sistem_initialization(); //initialize sistem (motors on 90 degre and lazer down)
  delay(500);
}

void loop() {
  if(Serial.available()){
  Receive = Serial.readStringUntil('\n'); // receive deta from serial port
  
  int n = Receive.length();
  char command[n + 1];
  command[n] = '\0';
  
  if (Receive.length() > 0){
    for(int i = 0;i<n;i++){
      command[i] = Receive[i];
    }
        command_x = command[0];
        command_y = command[2];
        command_l = command[4];
        
      }
      /*Serial.println(Receive.length());
      Serial.println(Receive);
      Serial.println(command_x);
      Serial.println(command_y);*/

      // control lazer 
      if(command_l == 'f'){
        lazer_fire();
      }
      else if(command_l == 's'){
        lazer_stop_fire();
      }
      else{
        lazer_stop_fire();
      }

      // control y axis motor
      if(command_y == 'u'){
        Serial.println(command_y);
        sistem_up();
      }
      else if(command_y == 'd'){
        sistem_down();
      }
      else{
        x_motor.write(angle_x);
      }

      // control x axis motor
      if(command_x == 'l'){
        sistem_left();
      }
      else if(command_x == 'r'){
        sistem_right();
      }
      else{
        y_motor.write(angle_y);
      }

      // initialize sistem when 'c' command is get
      if(command_x == 'c' && command_y =='c' && command_l == 'c'){
        sistem_initialization();
      }
}

}


void lazer_fire(){
  digitalWrite(lazer,fire);
}

void lazer_stop_fire(){
  
  digitalWrite(lazer,stop_fire);
}

void sistem_up(){
  angle_y = angle_y+increment_angle;
  if(angle_y >= max_angle){
    angle_y=max_angle;
  }
  y_motor.write(angle_y);
  
}

void sistem_down(){
  angle_y = angle_y-increment_angle;
  if(angle_y <= min_angle){
    angle_y=min_angle;
  }
  y_motor.write(angle_y);
  
}

void sistem_left(){
  angle_x = angle_x-increment_angle;
  if(angle_x >= max_angle){
    angle_x=max_angle;
  }
  x_motor.write(angle_x);
}

void sistem_right(){
  angle_x = angle_x+increment_angle;
  if(angle_x <= min_angle){
    angle_x=min_angle;
  }
  x_motor.write(angle_x);
}



void sistem_initialization(){
  x_motor.write(initialization_angle);
  y_motor.write(initialization_angle);
  digitalWrite(lazer,0);
}

void motor_debug(){
    for(int i=0; i< 179 ; i++){
    x_motor.write(i);
    delay(15);
    Serial.println(i);
  }
  
  for(int i=179; i>0 ; i--){
    x_motor.write(i);
    delay(15);
  }
}
