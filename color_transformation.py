import cv2
import numpy as np
from skimage import filters

class ColorTransformer:
    def __init__(self):
        pass
    
    def _apply_edge_preservation(self, image, mask, strength=0.5, method='sobel'):
        """
        Apply edge preservation to maintain eye details
        
        Args:
            image: Input image (BGR format)
            mask: Eye mask
            strength: Edge preservation strength (0-1)
            method: Edge detection method ('sobel', 'canny', 'prewitt')
            
        Returns:
            edge_mask: Edge mask for preserving details
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to focus on eye region
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Apply edge detection based on method
        if method == 'sobel':
            # Sobel edge detection
            sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mask = cv2.magnitude(sobelx, sobely)
            
        elif method == 'canny':
            # Canny edge detection
            edge_mask = cv2.Canny(masked_gray, 50, 150)
            
        elif method == 'prewitt':
            try:
                # Prewitt edge detection using scikit-image
                # Convert to float for scikit-image processing
                masked_gray_float = masked_gray.astype(float) / 255.0
                edge_mask = filters.prewitt(masked_gray_float)
                edge_mask = (edge_mask * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error in Prewitt edge detection: {e}")
                # Fall back to Sobel if Prewitt fails
                sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_mask = cv2.magnitude(sobelx, sobely)
            
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Normalize edge mask to 0-1 range
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Apply strength factor
        edge_mask = edge_mask * strength
        
        # Convert back to uint8 for visualization
        edge_mask = (edge_mask * 255).astype(np.uint8)
        
        return edge_mask
    
    def change_eye_color_rgb(self, image, eye_mask, target_color, intensity=0.7, blend_factor=0.5, edge_strength=0.5, edge_method='sobel'):
        """
        Change eye color using RGB color space
        
        Args:
            image: Input image (BGR format)
            eye_mask: Binary mask for the eye
            target_color: Target color in BGR format (B, G, R)
            intensity: Color intensity (0-1)
            blend_factor: Blending factor with original eye (0-1)
            edge_strength: Edge preservation strength (0-1)
            edge_method: Edge detection method
            
        Returns:
            result: Image with changed eye color
        """
        # Create a copy of the input image
        result = image.copy()
        
        # Get edge mask for detail preservation
        edge_mask = self._apply_edge_preservation(image, eye_mask, edge_strength, edge_method)
        
        # Create a color overlay with the target color
        color_overlay = np.zeros_like(image)
        color_overlay[:] = target_color
        
        # Extract the eye region from the original image
        eye_region = cv2.bitwise_and(image, image, mask=eye_mask)
        
        # Blend the original eye with the target color
        blended = cv2.addWeighted(eye_region, 1 - blend_factor, 
                                 cv2.bitwise_and(color_overlay, color_overlay, mask=eye_mask), 
                                 blend_factor * intensity, 0)
        
        # Apply edge preservation
        edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])
        blended = cv2.bitwise_and(blended, cv2.bitwise_not(edge_mask_3ch))
        edge_region = cv2.bitwise_and(eye_region, edge_mask_3ch)
        blended = cv2.add(blended, edge_region)
        
        # Create an inverse mask
        inverse_mask = cv2.bitwise_not(eye_mask)
        
        # Extract the background (non-eye region)
        background = cv2.bitwise_and(result, result, mask=inverse_mask)
        
        # Combine the background with the color-changed eye
        result = cv2.add(background, blended)
        
        return result
    
    def change_eye_color_hsv(self, image, eye_mask, target_color, intensity=0.7, blend_factor=0.5, edge_strength=0.5, edge_method='sobel'):
        """
        Change eye color using HSV color space
        
        Args:
            image: Input image (BGR format)
            eye_mask: Binary mask for the eye
            target_color: Target color in BGR format (B, G, R)
            intensity: Color intensity (0-1)
            blend_factor: Blending factor with original eye (0-1)
            edge_strength: Edge preservation strength (0-1)
            edge_method: Edge detection method
            
        Returns:
            result: Image with changed eye color
        """
        # Create a copy of the input image
        result = image.copy()
        
        # Get edge mask for detail preservation
        edge_mask = self._apply_edge_preservation(image, eye_mask, edge_strength, edge_method)
        
        # Convert the image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Convert target color to HSV
        target_color_bgr = np.uint8([[target_color]])
        target_color_hsv = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # Extract the eye region
        eye_region_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=eye_mask)
        
        # Create a modified eye region with the target hue and saturation
        modified_eye_hsv = eye_region_hsv.copy()
        
        # Only modify pixels within the eye mask
        mask_indices = np.where(eye_mask > 0)
        
        # Modify hue and saturation while preserving value (brightness)
        modified_eye_hsv[mask_indices[0], mask_indices[1], 0] = target_color_hsv[0]  # Hue
        modified_eye_hsv[mask_indices[0], mask_indices[1], 1] = np.clip(
            modified_eye_hsv[mask_indices[0], mask_indices[1], 1] * (1 - blend_factor) + 
            target_color_hsv[1] * blend_factor * intensity, 
            0, 255
        ).astype(np.uint8)  # Saturation
        
        # Convert back to BGR
        modified_eye_bgr = cv2.cvtColor(modified_eye_hsv, cv2.COLOR_HSV2BGR)
        
        # Apply edge preservation
        edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])
        modified_eye_bgr = cv2.bitwise_and(modified_eye_bgr, cv2.bitwise_not(edge_mask_3ch))
        edge_region = cv2.bitwise_and(cv2.bitwise_and(image, image, mask=eye_mask), edge_mask_3ch)
        modified_eye_bgr = cv2.add(modified_eye_bgr, edge_region)
        
        # Create an inverse mask
        inverse_mask = cv2.bitwise_not(eye_mask)
        
        # Extract the background (non-eye region)
        background = cv2.bitwise_and(result, result, mask=inverse_mask)
        
        # Combine the background with the color-changed eye
        result = cv2.add(background, modified_eye_bgr)
        
        return result
    
    def change_eye_color(self, image, left_eye_mask, right_eye_mask, target_color, 
                         color_space='hsv', intensity=0.7, blend_factor=0.5, 
                         edge_strength=0.5, edge_method='sobel'):
        """
        Change eye color for both eyes
        
        Args:
            image: Input image (BGR format)
            left_eye_mask: Binary mask for the left eye
            right_eye_mask: Binary mask for the right eye
            target_color: Target color in BGR format (B, G, R)
            color_space: Color space to use ('rgb' or 'hsv')
            intensity: Color intensity (0-1)
            blend_factor: Blending factor with original eye (0-1)
            edge_strength: Edge preservation strength (0-1)
            edge_method: Edge detection method
            
        Returns:
            result: Image with changed eye colors
        """
        # Process left eye
        if color_space == 'rgb':
            result = self.change_eye_color_rgb(
                image, left_eye_mask, target_color, 
                intensity, blend_factor, edge_strength, edge_method
            )
        else:  # hsv
            result = self.change_eye_color_hsv(
                image, left_eye_mask, target_color, 
                intensity, blend_factor, edge_strength, edge_method
            )
        
        # Process right eye
        if color_space == 'rgb':
            result = self.change_eye_color_rgb(
                result, right_eye_mask, target_color, 
                intensity, blend_factor, edge_strength, edge_method
            )
        else:  # hsv
            result = self.change_eye_color_hsv(
                result, right_eye_mask, target_color, 
                intensity, blend_factor, edge_strength, edge_method
            )
        
        return result 