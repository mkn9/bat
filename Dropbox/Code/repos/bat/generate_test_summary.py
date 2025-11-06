#!/usr/bin/env python3
"""
Generate a test summary report from test files
"""

import re
import os

def extract_test_cases(test_file):
    """Extract all test cases from a test file"""
    with open(test_file, 'r') as f:
        content = f.read()
    
    test_cases = []
    lines = content.split('\n')
    current_class = None
    
    for i, line in enumerate(lines):
        # Check for class definition
        class_match = re.match(r'^\s*class\s+(\w+).*:', line)
        if class_match:
            current_class = class_match.group(1)
        
        # Check for test function
        test_match = re.match(r'^\s*def\s+(test_\w+).*:', line)
        if test_match:
            test_name = test_match.group(1)
            # Get docstring if available
            docstring = ""
            if i + 1 < len(lines) and '"""' in lines[i + 1]:
                docstring = lines[i + 1].strip().replace('"""', '').strip()
            
            test_cases.append({
                'class': current_class or 'Function',
                'name': test_name,
                'docstring': docstring
            })
    
    return test_cases

def generate_summary():
    """Generate test summary report"""
    test_files = [
        'test_kinematics_formulas.py',
        'test_generate_kinematics_examples.py',
        'test_kinematics_to_video.py',
        'test_checkpoint_resume.py',
        'test_training_checkpoint_persistence.py'
    ]
    
    # Try to parse test results from XML if available
    test_results = {}
    if os.path.exists('test_results_all.xml'):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse('test_results_all.xml')
            root = tree.getroot()
            for testcase in root.iter('testcase'):
                test_name = testcase.get('name')
                class_name = testcase.get('classname', '').split('.')[-1]
                key = f"{class_name}::{test_name}" if class_name else test_name
                test_results[key] = len(list(testcase.iter('failure'))) == 0
        except Exception as e:
            pass  # If XML parsing fails, continue without results
    
    all_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            tests = extract_test_cases(test_file)
            all_tests.append((test_file, tests))
    
    # Generate markdown report
    report = "# Test Cases Summary\n\n"
    report += f"Generated: {os.popen('date').read().strip()}\n\n"
    report += f"**Total Test Cases: {sum(len(tests) for _, tests in all_tests)}**\n\n"
    
    for test_file, tests in all_tests:
        report += f"## {test_file}\n\n"
        report += f"**Total: {len(tests)} test cases**\n\n"
        
        # Group by class
        classes = {}
        for test in tests:
            class_name = test['class']
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(test)
        
        for class_name, class_tests in classes.items():
            report += f"### {class_name}\n\n"
            for test in class_tests:
                # Check if we have test results
                test_key = f"{class_name}::{test['name']}"
                status = ""
                if test_key in test_results:
                    status = " ✅ PASSED" if test_results[test_key] else " ❌ FAILED"
                
                report += f"- **{test['name']}**{status}\n"
                if test['docstring']:
                    report += f"  - {test['docstring']}\n"
            report += "\n"
    
    # Save report
    with open('test_summary.md', 'w') as f:
        f.write(report)
    
    print("✅ Test summary generated: test_summary.md")
    print(f"   Total test cases: {sum(len(tests) for _, tests in all_tests)}")

if __name__ == '__main__':
    generate_summary()

