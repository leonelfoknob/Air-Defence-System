#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>
#include <ESP32Servo.h>

/*
 * command map
 * 1 --> sistem up (x axis motor up)
 * -1 --> sistem down (x axis motor down)
 * 1 --> sistem left (y axis motor left)
 * -1 --> sistem right (y axis motor right)
 * 1 --> sistem fire (set lazer up for fire)
 * 0 --> sistem stop fire (set lazer down to stop fire)
 * -10 --> sistem initialization initialize sistem (motors on 90 degre and lazer down)
 * 0 all motors save last position
 */

Servo x_motor; // x axis motor
Servo y_motor; // y axis motor
int lazer = 13; // lazer pin

#define LAZER_PIN 13
#define X_AXIS_PIN 12
#define Y_AXIS_PIN 11

int initialization_angle = 90; // angle to set motor on 90 degre position
int max_angle = 175; // maximum angle that motor can reach 
int min_angle = 5; //manimum angle that motor can reach 
int angle_x=90; // angle to initialize x_motor to 90 degre
int angle_y=90; // angle to initialize y_motor to 90 degre
int increment_angle = 1; // angle that add or substitute on motor angle to turn it
int fire = 1; // value to set up lazer for fire
int stop_fire = 0; // value to set down lazer to stop fire

rcl_subscription_t lazer_subscriber;
rcl_subscription_t x_axis_subscriber;
rcl_subscription_t y_axis_subscriber;

std_msgs__msg__Int32 lazer_msg;
std_msgs__msg__Int32 x_axis_msg;
std_msgs__msg__Int32 y_axis_msg;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

void error_loop() {
    while (1) {
        digitalWrite(LAZER_PIN, !digitalRead(LAZER_PIN));
        delay(100);
    }
}

void lazer_callback(const void *msgin) {
    const std_msgs__msg__Int32 *msg = (const std_msgs__msg__Int32 *)msgin;
    //digitalWrite(LAZER_PIN, msg->data);
    // control lazer 
      if(msg->data == 1){
        lazer_fire();
      }
      else if(msg->data == 0){
        lazer_stop_fire();
      }
}

void x_axis_callback(const void *msgin) {
    const std_msgs__msg__Int32 *msg = (const std_msgs__msg__Int32 *)msgin;
    //digitalWrite(X_AXIS_PIN, msg->data);
    // control x axis motor
      if(msg->data == 1){
        sistem_left();
      }
      else if(msg->data == -1){
        sistem_right();
      }
      else if(msg->data == 0){
        y_motor.write(angle_y);
      }
      else if(msg->data == -10){
        sistem_initialization();
      }
      
}

void y_axis_callback(const void *msgin) {
    const std_msgs__msg__Int32 *msg = (const std_msgs__msg__Int32 *)msgin;
    digitalWrite(Y_AXIS_PIN, msg->data);
    // control y axis motor
      if(msg->data == 1){
        sistem_up();
      }
      else if(msg->data == -1){
        sistem_down();
      }
      else if(msg->data == 0){
        x_motor.write(angle_x);
      }
      else if(msg->data == -10){
        sistem_initialization();
      }
}

void setup() {
    set_microros_transports();

  
  x_motor.attach(14); //attach x axis motor on pin2 (pwm pin on arduino mega)
  y_motor.attach(12); //attach y axis motor on pin3 (pwm pin on arduino mega)
  pinMode(lazer,OUTPUT); // set lazer pin like outpu

  sistem_initialization(); //initialize sistem (motors on 90 degre and lazer down)
    digitalWrite(lazer, LOW);
    
    //delay(2000);
    delay(2000);
    
    allocator = rcl_get_default_allocator();
    RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
    RCCHECK(rclc_node_init_default(&node, "micro_ros_arduino_node", "", &support));
    
    RCCHECK(rclc_subscription_init_default(
        &lazer_subscriber, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "/lazer_command"));
    
    RCCHECK(rclc_subscription_init_default(
        &x_axis_subscriber, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "/x_axis_command"));
    
    RCCHECK(rclc_subscription_init_default(
        &y_axis_subscriber, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "/y_axis_command"));
    
    RCCHECK(rclc_executor_init(&executor, &support.context, 3, &allocator));
    RCCHECK(rclc_executor_add_subscription(&executor, &lazer_subscriber, &lazer_msg, &lazer_callback, ON_NEW_DATA));
    RCCHECK(rclc_executor_add_subscription(&executor, &x_axis_subscriber, &x_axis_msg, &x_axis_callback, ON_NEW_DATA));
    RCCHECK(rclc_executor_add_subscription(&executor, &y_axis_subscriber, &y_axis_msg, &y_axis_callback, ON_NEW_DATA));
}

void loop() {
    //delay(100);
    RCCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
}

//function 

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
