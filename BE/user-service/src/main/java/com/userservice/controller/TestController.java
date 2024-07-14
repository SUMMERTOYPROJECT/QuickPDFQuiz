package com.userservice.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {
    @GetMapping("/user-service/test")
    public String hello_world(){
        return "Hello World";
    }
}
